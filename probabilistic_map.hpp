#pragma once

#include "bonxai.hpp"
#include <Eigen/Geometry>
#include <unordered_set>

namespace Bonxai
{

template <class Functor>
void RayIterator(const CoordT& key_origin,
                 const CoordT& key_end,
                 const Functor& func);

inline void ComputeRay(const CoordT& key_origin,
                       const CoordT& key_end,
                       std::vector<CoordT>& ray)
{
  ray.clear();
  RayIterator(key_origin, key_end, [&ray](const CoordT& coord)
              {
                ray.push_back(coord);
                return true;
              } );
}

/**
 * @brief The ProbabilisticMap class is meant to behave as much as possible as
 * octomap::Octree, given the same voxel size.
 *
 * Insert a point cloud to update the current probability
 */
class ProbabilisticMap
{
public:
  using Vector3D = Eigen::Vector3d;

  /// Compute the logds, but return the result as an integer,
  /// The real number is represented as a fixed precision
  /// integer (6 decimals after the comma)
  [[nodiscard]] static constexpr int32_t logods(float prob)
  {
    return int32_t(1e6 * std::log(prob / (1.0 - prob)));
  }

  /// Expect the fixed comma value returned by logods()
  [[nodiscard]] static constexpr float prob(int32_t logods_fixed)
  {
    float logods = float(logods_fixed) * 1e-6;
    return (1.0 - 1.0 / (1.0 + std::exp(logods)));
  }

  struct CellT
  {
    // variable used to check if a cell was already updated in this loop
    int32_t update_id : 4;
    // the probability of the cell to be occupied
    int32_t probability_log : 28;
    // the previous probability of the cell to be occupied for tracking change
    int32_t prev_probability_free_log : 28;
    // the previous probability of the cell to be occupied for tracking change
    int32_t prev_probability_occupied_log : 28;

    CellT()
      : update_id(0)
      , probability_log(UnknownProbability)
      , prev_probability_occupied_log(UnknownProbability)
      , prev_probability_free_log(UnknownProbability){};
  };

  /// These default values are the same as OctoMap
  struct Options
  {
    int32_t prob_miss_log = logods(0.45f);
    int32_t prob_hit_log = logods(0.65f);

    int32_t clamp_min_log = logods(0.12f);
    int32_t clamp_max_log = logods(0.97f);

    int32_t occupancy_threshold_log = logods(0.5);
  };

  static const int32_t UnknownProbability;

  ProbabilisticMap(double resolution);

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
  void insertPointCloud(const std::vector<PointT, Allocator>& points,
                        const PointT& origin,
                        double max_range);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addHitPoint(const Vector3D& point);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addMissPoint(const Vector3D& point);

  [[nodiscard]] bool isOccupied(const Bonxai::CoordT& coord) const;

  [[nodiscard]] bool isUnknown(const Bonxai::CoordT& coord) const;

  [[nodiscard]] bool isFree(const Bonxai::CoordT& coord) const;

  void getOccupiedVoxels(std::vector<Bonxai::CoordT>& coords);

  void getFreeVoxels(std::vector<Bonxai::CoordT>& coords);

  void getNewFreeVoxelsAndReset(std::vector<Bonxai::CoordT>& coords);

  void getNewOccupiedVoxelsAndReset(std::vector<Bonxai::CoordT>& coords);

  template <typename PointT>
  void getOccupiedVoxels(std::vector<PointT>& points)
  {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getOccupiedVoxels(coords);
    for (const auto& coord : coords)
    {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }
  template <typename PointT>
  void getFreeVoxels(std::vector<PointT>& points)
  {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getFreeVoxels(coords);
    for (const auto& coord : coords)
    {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }
  template <typename PointT>
  void getNewFreeVoxelsAndReset(std::vector<PointT>& points)
  {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getNewFreeVoxelsAndReset(coords);
    for (const auto& coord : coords)
    {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }
  template <typename PointT>
  void getNewOccupiedVoxelsAndReset(std::vector<PointT>& points)
  {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getNewOccupiedVoxelsAndReset(coords);
    for (const auto& coord : coords)
    {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }

private:
  VoxelGrid<CellT> _grid;
  Options _options;
  uint8_t _update_count = 1;

  std::vector<CoordT> _miss_coords;
  std::vector<CoordT> _hit_coords;

  mutable Bonxai::VoxelGrid<CellT>::Accessor _accessor;

  void updateFreeCells(const Vector3D& origin);
};

//--------------------------------------------------

template <typename PointT, typename Alloc>
inline void ProbabilisticMap::insertPointCloud(const std::vector<PointT, Alloc>& points,
                                               const PointT& origin,
                                               double max_range)
{
  const auto from = ConvertPoint<Vector3D>(origin);
  const double max_range_sqr = max_range * max_range;
  for (const auto& point : points)
  {
    const auto to = ConvertPoint<Vector3D>(point);
    Vector3D vect(to - from);
    const double squared_norm = vect.squaredNorm();
    // points that exceed the max_range will create a cleaning ray
    if (squared_norm >= max_range_sqr)
    {
      // The new point will have distance == max_range from origin
      const Vector3D new_point = from + ((vect / std::sqrt(squared_norm)) * max_range);
      addMissPoint(new_point);
    }
    else {
      addHitPoint(to);
    }
  }
  updateFreeCells(from);
}

template <class Functor> inline
void RayIterator(const CoordT& key_origin,
                 const CoordT& key_end,
                 const Functor &func)
{
  if (key_origin == key_end)
  {
    return;
  }
  if(!func(key_origin))
  {
    return;
  }

  CoordT error = { 0, 0, 0 };
  CoordT coord = key_origin;
  CoordT delta = (key_end - coord);
  const CoordT step = { delta.x < 0 ? -1 : 1,
                        delta.y < 0 ? -1 : 1,
                        delta.z < 0 ? -1 : 1 };

  delta = { delta.x < 0 ? -delta.x : delta.x,
            delta.y < 0 ? -delta.y : delta.y,
            delta.z < 0 ? -delta.z : delta.z };

  const int max = std::max(std::max(delta.x, delta.y), delta.z);

  // maximum change of any coordinate
  for (int i = 0; i < max - 1; ++i)
  {
    // update errors
    error = error + delta;
    // manual loop unrolling
    if ((error.x << 1) >= max)
    {
      coord.x += step.x;
      error.x -= max;
    }
    if ((error.y << 1) >= max)
    {
      coord.y += step.y;
      error.y -= max;
    }
    if ((error.z << 1) >= max)
    {
      coord.z += step.z;
      error.z -= max;
    }
    if(!func(coord))
    {
      return;
    }
  }
}

}  // namespace Bonxai
















namespace Bonxai
{

const int32_t ProbabilisticMap::UnknownProbability = ProbabilisticMap::logods(0.5f);



VoxelGrid<ProbabilisticMap::CellT>& ProbabilisticMap::grid()
{
  return _grid;
}

ProbabilisticMap::ProbabilisticMap(double resolution)
  : _grid(resolution)
  , _accessor(_grid.createAccessor())
{}

const VoxelGrid<ProbabilisticMap::CellT>& ProbabilisticMap::grid() const
{
  return _grid;
}

const ProbabilisticMap::Options& ProbabilisticMap::options() const
{
  return _options;
}

void ProbabilisticMap::setOptions(const Options& options)
{
  _options = options;
}

void ProbabilisticMap::addHitPoint(const Vector3D &point)
{
  const auto coord = _grid.posToCoord(point);
  CellT* cell = _accessor.value(coord, true);

  if (cell->update_id != _update_count)
  {
    cell->probability_log = std::min(cell->probability_log + _options.prob_hit_log,
                                     _options.clamp_max_log);
    cell->prev_probability_free_log = cell->probability_log;

    cell->update_id = _update_count;
    _hit_coords.push_back(coord);
  }
}

void ProbabilisticMap::addMissPoint(const Vector3D &point)
{
  const auto coord = _grid.posToCoord(point);
  CellT* cell = _accessor.value(coord, true);

  if (cell->update_id != _update_count)
  {
    cell->probability_log = std::max(
        cell->probability_log + _options.prob_miss_log, _options.clamp_min_log);
    cell->prev_probability_occupied_log = cell->probability_log;

    cell->update_id = _update_count;
    _miss_coords.push_back(coord);
  }
}

bool ProbabilisticMap::isOccupied(const CoordT &coord) const
{
  if(auto* cell = _accessor.value(coord, false))
  {
    return cell->probability_log > _options.occupancy_threshold_log;
  }
  return false;
}

bool ProbabilisticMap::isUnknown(const CoordT &coord) const
{
  if(auto* cell = _accessor.value(coord, false))
  {
    return cell->probability_log == _options.occupancy_threshold_log;
  }
  return true;
}

bool ProbabilisticMap::isFree(const CoordT &coord) const
{
  if(auto* cell = _accessor.value(coord, false))
  {
    return cell->probability_log < _options.occupancy_threshold_log;
  }
  return false;
}

void Bonxai::ProbabilisticMap::updateFreeCells(const Vector3D& origin)
{
  auto accessor = _grid.createAccessor();

  // same as addMissPoint, but using lambda will force inlining
  auto clearPoint = [this, &accessor](const CoordT& coord)
  {
    CellT* cell = accessor.value(coord, true);
    if (cell->update_id != _update_count)
    {
      cell->probability_log = std::max(
          cell->probability_log + _options.prob_miss_log, _options.clamp_min_log);
      cell->prev_probability_occupied_log = cell->probability_log;
      cell->update_id = _update_count;
    }
    return true;
  };

  const auto coord_origin = _grid.posToCoord(origin);

  for (const auto& coord_end : _hit_coords)
  {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _hit_coords.clear();

  for (const auto& coord_end : _miss_coords)
  {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _miss_coords.clear();

  if (++_update_count == 4)
  {
    _update_count = 1;
  }
}

void ProbabilisticMap::getOccupiedVoxels(std::vector<CoordT>& coords)
{
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log > _options.occupancy_threshold_log)
    {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
}

void ProbabilisticMap::getFreeVoxels(std::vector<CoordT>& coords)
{
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log < _options.occupancy_threshold_log)
    {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
}

void ProbabilisticMap::getNewFreeVoxelsAndReset(std::vector<CoordT>& coords)
{
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log < _options.occupancy_threshold_log) // now free
    {
      if (cell.prev_probability_free_log >= _options.occupancy_threshold_log) // but, not free before
      {
        cell.prev_probability_free_log = cell.probability_log;
        coords.push_back(coord);
      }
    }
  };
  _grid.forEachCell(visitor);
}

void ProbabilisticMap::getNewOccupiedVoxelsAndReset(std::vector<CoordT>& coords)
{
  coords.clear();
  auto visitor = [&](CellT& cell, const CoordT& coord) {
    if (cell.probability_log > _options.occupancy_threshold_log) // now occupied
    {
      if (cell.prev_probability_occupied_log <= _options.occupancy_threshold_log) // but, not occupied before
      {
        cell.prev_probability_occupied_log = cell.probability_log;
        coords.push_back(coord);
      }
    }
  };
  _grid.forEachCell(visitor);
}

}  // namespace Bonxai