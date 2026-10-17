#pragma once

#include <Eigen/Geometry>
#include <unordered_map>
#include <utility>

#include "bonxai.hpp"

namespace Bonxai {
/**
 * @brief The ProbabilisticMapLio class is meant to behave as much as possible as
 * octomap::Octree, given the same voxel size.
 *
 * Insert a point cloud to update the current probability
 */
class ProbabilisticMapLio {
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
    int32_t prev_probability_free_log : 28;

    CellT()
        : update_id(0),
          probability_log(UnknownProbability),
          prev_probability_free_log(UnknownProbability){};
  };

  /// These default values are the same as OctoMap
  struct Options {
    int32_t prob_miss_log = logods(0.4f);
    int32_t prob_hit_log = logods(0.97f);

    int32_t clamp_min_log = logods(0.12f);
    int32_t clamp_max_log = logods(0.97f);

    int32_t occupancy_threshold_log = logods(0.5);

    // Applies to insertPointCloud()/updateFreeCells(); the legacy radius query
    // retains its original Bresenham traversal.
    enum class RayMode : uint8_t { Exact, Approximate };
    RayMode ray_mode = RayMode::Approximate;
  };

  static const int32_t UnknownProbability;

  ProbabilisticMapLio(double resolution);

  ProbabilisticMapLio(ProbabilisticMapLio&& other)
      : _grid(std::move(other._grid)),
        _options(other._options),
        _ray_targets(std::move(other._ray_targets)),
        _traversed_cells(std::move(other._traversed_cells)),
        _miss_before(std::move(other._miss_before)),
        _accessor(_grid.createAccessor()) {}

  ProbabilisticMapLio& operator=(ProbabilisticMapLio&& other) {
    if (this != &other) {
      _grid = std::move(other._grid);
      _options = other._options;
      _ray_targets = std::move(other._ray_targets);
      _traversed_cells = std::move(other._traversed_cells);
      _miss_before = std::move(other._miss_before);
    }
    return *this;
  }

  // Discard pending endpoints as well as the grid, so cached cell pointers
  // cannot outlive their allocation.
  void Clear() {
    _ray_targets.clear();
    _traversed_cells.clear();
    _miss_before.clear();
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
                        double max_range);

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

  [[nodiscard]] bool isOccupied(const CoordT& coord) const;

  [[nodiscard]] bool isUnknown(const CoordT& coord) const;

  [[nodiscard]] bool isFree(const CoordT& coord) const;

  void getOccupiedVoxels(std::vector<CoordT>& coords);

  void getFreeVoxels(std::vector<CoordT>& coords);

  void getNewFreeVoxelsAndReset(std::vector<CoordT>& coords);

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
  void getNewFreeVoxelsAndReset(std::vector<PointT>& points) {
    thread_local std::vector<CoordT> coords;
    coords.clear();
    getNewFreeVoxelsAndReset(coords);
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
  std::vector<CellT*> _traversed_cells;

  // addHitPoint/addMissPoint keep updating immediately, as in this fork.
  // Snapshot miss endpoints so a later hit in the same scan can replace the
  // miss exactly, even if the miss hit the lower clamp. Hits always win.
  std::unordered_map<CellT*, CellT> _miss_before;

  mutable Bonxai::VoxelGrid<CellT>::Accessor _accessor;
};

//--------------------------------------------------

template <typename PointT, typename Alloc>
inline void ProbabilisticMapLio::insertPointCloud(const std::vector<PointT, Alloc>& points,
                                                  const PointT& origin, double max_range) {
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
inline const int32_t ProbabilisticMapLio::UnknownProbability = ProbabilisticMapLio::logods(0.5f);

inline VoxelGrid<ProbabilisticMapLio::CellT>& ProbabilisticMapLio::grid() { return _grid; }

inline ProbabilisticMapLio::ProbabilisticMapLio(double resolution)
    : _grid(resolution), _accessor(_grid.createAccessor()) {}

inline const VoxelGrid<ProbabilisticMapLio::CellT>& ProbabilisticMapLio::grid() const {
  return _grid;
}

inline const ProbabilisticMapLio::Options& ProbabilisticMapLio::options() const { return _options; }

inline void ProbabilisticMapLio::setOptions(const Options& options) {
  _options = options;
  return;
}

inline void ProbabilisticMapLio::addHitPoint(const Vector3D& point) {
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
  cell->prev_probability_free_log = cell->probability_log;
  cell->update_id = CellT::kHit;
}

inline void ProbabilisticMapLio::addMissPoint(const Vector3D& point) {
  const auto coord = _grid.posToCoord(point);
  CellT* cell = _accessor.value(coord, true);
  if (cell->update_id == CellT::kUnseen) {
    _miss_before.emplace(cell, *cell);
    _ray_targets.emplace_back(coord, cell);
    cell->probability_log =
        std::max(cell->probability_log + _options.prob_miss_log, _options.clamp_min_log);
    cell->update_id = CellT::kMiss;
  }
}

inline bool ProbabilisticMapLio::isOccupied(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->probability_log > _options.occupancy_threshold_log;
  }
  return false;
}

inline bool ProbabilisticMapLio::isUnknown(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->probability_log == _options.occupancy_threshold_log;
  }
  return true;
}

inline bool ProbabilisticMapLio::isFree(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->probability_log < _options.occupancy_threshold_log;
  }
  return false;
}

inline void ProbabilisticMapLio::updateFreeCells(const Vector3D& origin) {
  auto accessor = _grid.createAccessor();
  const auto options = _options;
  auto clearPoint = [this, &accessor, &options](const CoordT& coord) {
    CellT* cell = accessor.value(coord, true);
    if (cell->update_id == CellT::kUnseen) {
      _traversed_cells.push_back(cell);
      cell->probability_log =
          std::max(cell->probability_log + options.prob_miss_log, options.clamp_min_log);
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
    target.second->update_id = CellT::kUnseen;
  }
  for (auto* cell : _traversed_cells) {
    cell->update_id = CellT::kUnseen;
  }
  _ray_targets.clear();
  _traversed_cells.clear();
  _miss_before.clear();
}

inline void ProbabilisticMapLio::getOccupiedVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log > _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
  return;
}

inline void ProbabilisticMapLio::getFreeVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log < _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
  return;
}

inline void ProbabilisticMapLio::getNewFreeVoxelsAndReset(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log < _options.occupancy_threshold_log)  // now free
    {
      if (cell.prev_probability_free_log >=
          _options.occupancy_threshold_log)  // but, not free before
      {
        cell.prev_probability_free_log = cell.probability_log;
        coords.push_back(coord);
      }
    }
  };
  _grid.forEachCell(visitor);
  return;
}

}  // namespace Bonxai
