#pragma once

#include <Eigen/Geometry>
#include <unordered_map>
#include <utility>

#include "bonxai.hpp"

namespace Bonxai {

/**
 * @brief The ProbabilisticMap class is meant to behave as much as possible as
 * octomap::Octree, given the same voxel size.
 *
 * Insert a point cloud to update the current probability
 */
class ProbabilisticMap {
 public:
  using Vector3D = Eigen::Vector3d;

  /// Compute the logds, but return the result as an integer,
  /// The real number is represented as a fixed precision
  /// integer (6 decimals after the comma)
  [[nodiscard]] static constexpr int32_t logods(float prob) {
    if (prob == 0.5f) {
      return 0;
    }
    return int32_t(1e6 * std::log(prob / (1.0 - prob)));
  }
  /// Expect the fixed comma value returned by logods()
  [[nodiscard]] static constexpr float prob(int32_t logods_fixed) {
    if (logods_fixed == 0) {
      return 0.5f;
    }
    float logods = float(logods_fixed) * 1e-6;
    return (1.0 - 1.0 / (1.0 + std::exp(logods)));
  }

  struct CellT {
    // Transient scan state. Keep the original field name for source compatibility.
    enum : int32_t { kUnseen = 0, kMiss = 1, kHit = 2 };
    int32_t update_id : 4;
    // the probability of the cell to be occupied
    int32_t probability_log : 28;
    // the previous probability of the cell to be occupied for tracking change
    int32_t prev_probability_occupied_log : 28;
    int32_t queued_occupied : 2;
    // the previous probability of the cell to be occupied for tracking change
    int32_t prev_probability_free_log : 28;
    int32_t queued_free : 2;

    CellT()
        : update_id(0),
          probability_log(UnknownProbability),
          prev_probability_occupied_log(UnknownProbability),
          queued_occupied(0),
          prev_probability_free_log(UnknownProbability),
          queued_free(0){};
  };

  /// These default values are the same as OctoMap
  struct Options {
    int32_t prob_miss_log = logods(0.45f);
    int32_t prob_hit_log = logods(0.65f);

    int32_t clamp_min_log = logods(0.12f);
    int32_t clamp_max_log = logods(0.97f);

    int32_t occupancy_threshold_log = logods(0.5);

    // Applies to insertPointCloud()/updateFreeCells(); the legacy radius query
    // retains its original Bresenham traversal.
    enum class RayMode : uint8_t { Exact, Approximate };
    RayMode ray_mode = RayMode::Approximate;
  };

  static const int32_t UnknownProbability;

  ProbabilisticMap(const double& resolution);

  ProbabilisticMap(ProbabilisticMap&& other)
      : _grid(std::move(other._grid)),
        _options(other._options),
        _ray_targets(std::move(other._ray_targets)),
        _traversed_cells(std::move(other._traversed_cells)),
        _miss_before(std::move(other._miss_before)),
        _pending_free(std::move(other._pending_free)),
        _pending_occupied(std::move(other._pending_occupied)),
        _external_grid_access(other._external_grid_access),
        _accessor(_grid.createAccessor()) {}

  ProbabilisticMap& operator=(ProbabilisticMap&& other) {
    if (this != &other) {
      _grid = std::move(other._grid);
      _options = other._options;
      _ray_targets = std::move(other._ray_targets);
      _traversed_cells = std::move(other._traversed_cells);
      _miss_before = std::move(other._miss_before);
      _pending_free = std::move(other._pending_free);
      _pending_occupied = std::move(other._pending_occupied);
      _external_grid_access = _external_grid_access || other._external_grid_access;
    }
    return *this;
  }

  // Discard pending endpoints as well as the grid, so cached cell pointers
  // cannot outlive their allocation.
  void Clear() {
    _ray_targets.clear();
    _traversed_cells.clear();
    _miss_before.clear();
    _pending_free.clear();
    _pending_occupied.clear();
    _grid.Clear();
  }

  [[nodiscard]] VoxelGrid<CellT>& grid();
  [[nodiscard]] const VoxelGrid<CellT>& grid() const;

  [[nodiscard]] const Options& options() const;
  void setOptions(const Options& options);

  /**
   * @brief insertPointCloud will update the probability map
   * with a new set of detections.
   * The template function can accept points of different types,
   * such as pcl:Point, Eigen::Vector or Bonxai::Point3d
   *
   * Both origin and points must be in word coordinates
   *
   * @param points   a vector of points which represent detected obstacles
   * @param origin   origin of the point cloud
   * @param max_range  max range of the ray, if exceeded, we will use that
   * to compute a free space
   */
  template <typename PointT, typename Allocator>
  void insertPointCloud(const std::vector<PointT, Allocator>& points, const PointT& origin,
                        const double& max_range);
  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addHitPoint(const Vector3D& point);
  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addMissPoint(const Vector3D& point);

  // Complete one scan and reset only the cells touched by that scan.
  // Call after manually adding endpoints; insertPointCloud calls it for you.
  void updateFreeCells(const Vector3D& origin);

  /**
   * @brief raycast along the direction around the collision radius from the origin
   */
  template <typename PointT>
  void rayCastPointRadius(const PointT& origin_pt, const PointT& direction, const double& max_range,
                          const double& collision_radius, std::vector<PointT>& casted_points,
                          double& valid_ratio);

  [[nodiscard]] bool isOccupied(const CoordT& coord) const;
  [[nodiscard]] bool isUnknown(const CoordT& coord) const;
  [[nodiscard]] bool isFree(const CoordT& coord) const;

  void getOccupiedVoxels(std::vector<CoordT>& coords);
  void getFreeVoxels(std::vector<CoordT>& coords);
  void getNewFreeVoxelsAndReset(std::vector<CoordT>& coords,
                              std::vector<CoordT>* removed_occupied = nullptr);
  void getNewOccupiedVoxelsAndReset(std::vector<CoordT>& coords);

  template <typename PointT>
  void getOccupiedVoxels(std::vector<PointT>& points) {
    thread_local std::vector<CoordT> coords;
    coords.clear();
    getOccupiedVoxels(coords);
    for (const auto& coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
    return;
  }
  template <typename PointT>
  void getFreeVoxels(std::vector<PointT>& points) {
    thread_local std::vector<CoordT> coords;
    coords.clear();
    getFreeVoxels(coords);
    for (const auto& coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
    return;
  }
  template <typename PointT>
  void getNewFreeVoxelsAndReset(std::vector<PointT>& points,
                              std::vector<PointT>* removed_occupied = nullptr) {
    thread_local std::vector<CoordT> coords;
    thread_local std::vector<CoordT> removed;
    coords.clear();
    getNewFreeVoxelsAndReset(coords, removed_occupied ? &removed : nullptr);
    for (const auto& coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
    if (removed_occupied) {
      removed_occupied->clear();
      for (const auto& coord : removed) {
        const auto p = _grid.coordToPos(coord);
        removed_occupied->emplace_back(p.x, p.y, p.z);
      }
    }
    return;
  }
  template <typename PointT>
  void getNewFreeVoxelsAndReset(std::vector<PointT>& points, std::vector<CoordT>& coords) {
    coords.clear();
    getNewFreeVoxelsAndReset(coords);
    for (const auto& coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
    return;
  }
  template <typename PointT>
  void getNewOccupiedVoxelsAndReset(std::vector<PointT>& points) {
    thread_local std::vector<CoordT> coords;
    coords.clear();
    getNewOccupiedVoxelsAndReset(coords);
    for (const auto& coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
    return;
  }

 private:
  VoxelGrid<CellT> _grid;
  Options _options;
  std::vector<std::pair<CoordT, CellT*>> _ray_targets;
  std::vector<std::pair<CoordT, CellT*>> _traversed_cells;
  std::vector<std::pair<CoordT, CellT*>> _pending_free;
  std::vector<std::pair<CoordT, CellT*>> _pending_occupied;
  //Codex: A retained mutable grid reference can bypass observation tracking.
  bool _external_grid_access = false;

  void trackTransition(const CoordT& coord, CellT* cell) {
    const auto threshold = _options.occupancy_threshold_log;
    if (cell->probability_log < threshold && cell->prev_probability_free_log >= threshold && !cell->queued_free) {
      cell->queued_free = 1;
      _pending_free.emplace_back(coord, cell);
    }
    if (cell->probability_log > threshold && cell->prev_probability_occupied_log <= threshold && !cell->queued_occupied) {
      cell->queued_occupied = 1;
      _pending_occupied.emplace_back(coord, cell);
    }
  }

  // addHitPoint/addMissPoint keep updating immediately, as in this fork.
  // Snapshot miss endpoints so a later hit in the same scan can replace the
  // miss exactly, even if the miss hit the lower clamp. Hits always win.
  std::unordered_map<CellT*, CellT> _miss_before;

  mutable Bonxai::VoxelGrid<CellT>::Accessor _accessor;

  template <typename PointT>
  bool rayCastPoint(const CoordT& origin, const CoordT& end, PointT& casted_point);
};

//--------------------------------------------------

template <typename PointT, typename Alloc>
inline void ProbabilisticMap::insertPointCloud(const std::vector<PointT, Alloc>& points,
                                               const PointT& origin, const double& max_range) {
  const auto from = ConvertPoint<Vector3D>(origin);
  const double max_range_sqr = max_range * max_range;
  for (const auto& point : points) {
    const auto to = ConvertPoint<Vector3D>(point);
    Vector3D vect(to - from);
    const double squared_norm = vect.squaredNorm();
    // points that exceed the max_range will create a cleaning ray
    if (squared_norm >= max_range_sqr) {
      // The new point will have distance == max_range from origin
      const Vector3D new_point = from + ((vect / std::sqrt(squared_norm)) * max_range);
      addMissPoint(new_point);
    } else {
      addHitPoint(to);
    }
  }
  updateFreeCells(from);
}

}  // namespace Bonxai

namespace Bonxai {
inline const int32_t ProbabilisticMap::UnknownProbability = ProbabilisticMap::logods(0.5f);

inline VoxelGrid<ProbabilisticMap::CellT>& ProbabilisticMap::grid() {
  _external_grid_access = true;
  return _grid;
}

inline ProbabilisticMap::ProbabilisticMap(const double& resolution)
    : _grid(resolution), _accessor(_grid.createAccessor()) {}

inline const VoxelGrid<ProbabilisticMap::CellT>& ProbabilisticMap::grid() const { return _grid; }

inline const ProbabilisticMap::Options& ProbabilisticMap::options() const { return _options; }

inline void ProbabilisticMap::setOptions(const Options& options) {
  if (_options.occupancy_threshold_log != options.occupancy_threshold_log) {
    _external_grid_access = true;
  }
  _options = options;
  return;
}

inline void ProbabilisticMap::addHitPoint(const Vector3D& point) {
  const auto coord = _grid.posToCoord(point);
  CellT* cell = _accessor.value(coord, true);
  if (cell->update_id == CellT::kHit) {
    return;
  }
  if (cell->update_id == CellT::kUnseen) {
    _ray_targets.emplace_back(coord, cell);
  } else {
    // Restore the pre-miss probability and change-tracking state before the hit.
    *cell = _miss_before.at(cell);
    _miss_before.erase(cell);
  }
  cell->probability_log =
      std::min(cell->probability_log + _options.prob_hit_log, _options.clamp_max_log);
  //Codex: A weak hit while still free must not erase a pending occupied-to-free event.
  cell->prev_probability_free_log = std::max(cell->prev_probability_free_log, cell->probability_log);
  cell->update_id = CellT::kHit;
}

inline void ProbabilisticMap::addMissPoint(const Vector3D& point) {
  const auto coord = _grid.posToCoord(point);
  CellT* cell = _accessor.value(coord, true);
  if (cell->update_id == CellT::kUnseen) {
    _miss_before.emplace(cell, *cell);
    _ray_targets.emplace_back(coord, cell);
    cell->probability_log =
        std::max(cell->probability_log + _options.prob_miss_log, _options.clamp_min_log);
    cell->prev_probability_occupied_log = std::min(cell->prev_probability_occupied_log, cell->probability_log);
    cell->update_id = CellT::kMiss;
  }
}

inline bool ProbabilisticMap::isOccupied(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->probability_log > _options.occupancy_threshold_log;
  }
  return false;
}

inline bool ProbabilisticMap::isUnknown(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->probability_log == _options.occupancy_threshold_log;
  }
  return true;
}

inline bool ProbabilisticMap::isFree(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->probability_log < _options.occupancy_threshold_log;
  }
  return false;
}

inline void ProbabilisticMap::updateFreeCells(const Vector3D& origin) {
  auto accessor = _grid.createAccessor();
  const auto options = _options;
  auto clearPoint = [this, &accessor, &options](const CoordT& coord) {
    CellT* cell = accessor.value(coord, true);
    if (cell->update_id == CellT::kUnseen) {
      _traversed_cells.emplace_back(coord, cell);
      cell->probability_log =
          std::max(cell->probability_log + options.prob_miss_log, options.clamp_min_log);
      cell->prev_probability_occupied_log = std::min(cell->prev_probability_occupied_log, cell->probability_log);
      cell->update_id = CellT::kMiss;
    }
    return true;
  };

  const auto coord_origin = _grid.posToCoord(origin);
  for (const auto& target : _ray_targets) {
    if (options.ray_mode == Options::RayMode::Exact) {
      ExactRayIterator(origin, coord_origin, target.first, _grid.voxelSize(), clearPoint);
    } else {
      RayIterator(coord_origin, target.first, clearPoint);
    }
  }
  // No wrapping scan counter: cells are always unseen again between scans.
  for (const auto& target : _ray_targets) {
    trackTransition(target.first, target.second);
    target.second->update_id = CellT::kUnseen;
  }
  for (const auto& item : _traversed_cells) {
    trackTransition(item.first, item.second);
    item.second->update_id = CellT::kUnseen;
  }
  _ray_targets.clear();
  _traversed_cells.clear();
  _miss_before.clear();
}

inline void ProbabilisticMap::getOccupiedVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log > _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
  return;
}

inline void ProbabilisticMap::getFreeVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log < _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
  return;
}

inline void ProbabilisticMap::getNewFreeVoxelsAndReset(std::vector<CoordT>& coords,
                                                    std::vector<CoordT>* removed_occupied) {
  coords.clear();
  if (removed_occupied) {
    removed_occupied->clear();
  }
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    cell.queued_free = 0;
    if (cell.probability_log < _options.occupancy_threshold_log)  // now free
    {
      if (cell.prev_probability_free_log >=
          _options.occupancy_threshold_log)  // but, not free before
      {
        //Codex: Newly observed free space does not remove an obstacle witness.
        //Codex: Preserve the pre-consumption state, including delayed free consumption.
        if (removed_occupied && (_external_grid_access || cell.prev_probability_free_log > _options.occupancy_threshold_log)) {
          removed_occupied->push_back(coord);
        }
        cell.prev_probability_free_log = cell.probability_log;
        coords.push_back(coord);
      }
    }
  };
  if (_external_grid_access) {
    _grid.forEachCell(visitor);
  } else {
    for (const auto& item : _pending_free) {
      visitor(*item.second, item.first);
    }
  }
  _pending_free.clear();
  return;
}

inline void ProbabilisticMap::getNewOccupiedVoxelsAndReset(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    cell.queued_occupied = 0;
    if (cell.probability_log > _options.occupancy_threshold_log)  // now occupied
    {
      if (cell.prev_probability_occupied_log <=
          _options.occupancy_threshold_log)  // but, not occupied before
      {
        cell.prev_probability_occupied_log = cell.probability_log;
        coords.push_back(coord);
      }
    }
  };
  if (_external_grid_access) {
    _grid.forEachCell(visitor);
  } else {
    for (const auto& item : _pending_occupied) {
      visitor(*item.second, item.first);
    }
  }
  _pending_occupied.clear();
  return;
}

template <typename PointT>
inline bool ProbabilisticMap::rayCastPoint(const CoordT& origin, const CoordT& end,
                                           PointT& casted_point) {
  bool hit = false;
  RayIterator(origin, end, [&](const CoordT& coord) {
    if (coord != origin && isOccupied(coord)) {
      casted_point = ConvertPoint<PointT>(_grid.coordToPos(coord));
      hit = true;
      return false;
    }
    return true;
  });
  return hit;
}

template <typename PointT>
inline void ProbabilisticMap::rayCastPointRadius(const PointT& origin_pt, const PointT& direction,
                                                 const double& max_range,
                                                 const double& collision_radius,
                                                 std::vector<PointT>& casted_points,
                                                 double& valid_ratio) {
  casted_points.clear();
  const Vector3D origin_vec3d_ = ConvertPoint<Vector3D>(origin_pt);
  const CoordT origin_coord_ = _grid.posToCoord(origin_vec3d_);
  const Vector3D direction_vec3d_ = ConvertPoint<Vector3D>(direction).normalized() * max_range;
  const CoordT direction_coord_ = _grid.posToCoord(direction_vec3d_);
  const Vector3D min_bound_ = origin_vec3d_.array() - collision_radius;
  const Vector3D max_bound_ = origin_vec3d_.array() + collision_radius;
  const CoordT min_bound_coord_ = _grid.posToCoord(min_bound_);
  const CoordT max_bound_coord_ = _grid.posToCoord(max_bound_);

  int counter_ = 0;
  for (int32_t i = min_bound_coord_.x; i < max_bound_coord_.x + 1; i++) {
    for (int32_t j = min_bound_coord_.y; j < max_bound_coord_.y + 1; j++) {
      counter_++;
      PointT casted_point_;
      CoordT radius_origin_coord_ = {i, j, origin_coord_.z};
      CoordT end_coord_ = (radius_origin_coord_ + direction_coord_);
      if (rayCastPoint(radius_origin_coord_, end_coord_, casted_point_)) {
        casted_points.push_back(casted_point_);
      }
    }
  }
  if (counter_ == 0) {
    valid_ratio = 0.0;
  } else {
    valid_ratio = casted_points.size() / (double)counter_;
  }
  return;
}

}  // namespace Bonxai
