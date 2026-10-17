/*
 * Copyright Contributors to the Bonxai Project
 * Copyright Contributors to the OpenVDB Project
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

// Core amalgamated from facontidavide/Bonxai, commit
// 8ba04b20f34da1d066db48b88313be101e6ee10e (2026-09-28).
// Fork compatibility: runtime grid shapes, legacy public APIs, voxel centres,
// and shared ray helpers for the two standalone probabilistic-map headers.

// ---- bonxai.hpp ----

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// ---- coord_map.hpp ----

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ---- grid_coord.hpp ----

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Bonxai {

// Magically converts any representation of a point in 3D
// (type with x, y and z) to another one. Works with:
//
// - pcl::PointXYZ, pcl::PointXYZI, pcl::PointXYZRGB, etc
// - Eigen::Vector3d, Eigen::Vector3f
// - custom type with x,y,z fields. In this case Bonxai::Point3D
// - arrays or vectors with 3 elements.

template <typename PointOut, typename PointIn>
PointOut ConvertPoint(const PointIn& v);

struct Point3D {
  double x;
  double y;
  double z;

  Point3D() = default;

  Point3D(const Point3D& v) = default;
  Point3D(Point3D&& v) = default;

  Point3D& operator=(const Point3D& v) = default;
  Point3D& operator=(Point3D&& v) = default;

  Point3D(double x, double y, double z);

  template <typename T>
  Point3D(const T& v) {
    *this = ConvertPoint<Point3D>(v);
  }

  template <typename T>
  Point3D& operator=(const T& v) {
    *this = ConvertPoint<Point3D>(v);
    return *this;
  }

  // Access to x, y, z, using index 0, 1, 2
  [[nodiscard]] double& operator[](size_t index);
};

struct CoordT {
  int32_t x;
  int32_t y;
  int32_t z;

  // Access to x, y, z, using index 0, 1, 2
  [[nodiscard]] int32_t& operator[](size_t index);

  [[nodiscard]] bool operator==(const CoordT& other) const;
  [[nodiscard]] bool operator!=(const CoordT& other) const;

  [[nodiscard]] CoordT operator+(const CoordT& other) const;
  [[nodiscard]] CoordT operator-(const CoordT& other) const;

  CoordT& operator+=(const CoordT& other);
  CoordT& operator-=(const CoordT& other);
};

[[nodiscard]] inline CoordT PosToCoord(const Point3D& point, double inv_resolution) {
  return {static_cast<int32_t>(std::floor(point.x * inv_resolution)),
          static_cast<int32_t>(std::floor(point.y * inv_resolution)),
          static_cast<int32_t>(std::floor(point.z * inv_resolution))};
}

[[nodiscard]] inline Point3D CoordToPos(const CoordT& coord, double resolution) {
  return {(static_cast<double>(coord.x) + 0.5) * resolution,
          (static_cast<double>(coord.y) + 0.5) * resolution,
          (static_cast<double>(coord.z) + 0.5) * resolution};
}

//----------------------------------------------------
//----------------- Implementations ------------------
//----------------------------------------------------

inline Point3D::Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

inline double& Point3D::operator[](size_t index) {
  switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    default:
      throw std::runtime_error("out of bound index");
  }
}

// clang-format off
template <class T, class = void>
struct type_has_method_x : std::false_type {};
template <class T>
struct type_has_method_x<T, std::void_t<decltype(std::declval<const T&>().x())>> : std::true_type {};

template <class T, class = void>
struct type_has_member_x : std::false_type {};
template <class T>
struct type_has_member_x<T, std::void_t<decltype(T::x)>> : std::true_type {};

template<typename>
struct type_is_vector : std::false_type {};
template<typename T, typename A>
struct type_is_vector<std::vector<T, A>> : std::true_type {};
template<typename T>
struct type_is_vector<std::array<T, 3>> : std::true_type {};
// clang-format on

template <typename PointOut, typename PointIn>
inline PointOut ConvertPoint(const PointIn& v) {
  // clang-format off
  static_assert(std::is_same_v<PointIn, PointOut> ||
                    type_has_method_x<PointIn>::value ||
                    type_has_member_x<PointIn>::value ||
                    type_is_vector<PointIn>::value,
                "Can't convert from the specified type");

  static_assert(std::is_same_v<PointIn, PointOut> ||
                    type_has_method_x<PointOut>::value ||
                    type_has_member_x<PointOut>::value ||
                    type_is_vector<PointOut>::value,
                "Can't convert to the specified type");

  // clang-format on
  if constexpr (std::is_same_v<PointIn, PointOut>) {
    return v;
  }
  if constexpr (type_has_method_x<PointIn>::value) {
    return {v.x(), v.y(), v.z()};
  }
  if constexpr (type_has_member_x<PointIn>::value) {
    return {v.x, v.y, v.z};
  }
  if constexpr (type_is_vector<PointIn>::value) {
    return {v[0], v[1], v[2]};
  }
}

inline int32_t& CoordT::operator[](size_t index) {
  switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    default:
      throw std::runtime_error("out of bound index");
  }
}

inline bool CoordT::operator==(const CoordT& other) const {
  return x == other.x && y == other.y && z == other.z;
}

inline bool CoordT::operator!=(const CoordT& other) const { return !(*this == other); }

inline CoordT CoordT::operator+(const CoordT& other) const {
  return {x + other.x, y + other.y, z + other.z};
}

inline CoordT CoordT::operator-(const CoordT& other) const {
  return {x - other.x, y - other.y, z - other.z};
}

inline CoordT& CoordT::operator+=(const CoordT& other) {
  x += other.x;
  y += other.y;
  z += other.z;
  return *this;
}

inline CoordT& CoordT::operator-=(const CoordT& other) {
  x -= other.x;
  y -= other.y;
  z -= other.z;
  return *this;
}

}  // namespace Bonxai

namespace std {
template <>
struct hash<Bonxai::CoordT> {
  std::size_t operator()(const Bonxai::CoordT& p) const {
    // For the containers of the users: VoxelGrid's root map, a CoordMap, has its own hash.
    //
    // Each coordinate is multiplied by its own odd constant and murmur3's finalizer mixes
    // the lot: every bit of the result is good, whatever the container does with them.
    //
    // The coordinates are read one at a time on purpose. Packed as x | y << 32, they are
    // read with one 64 bits load, which cannot be forwarded from the 32 bits stores that
    // write a CoordT computed field by field: when such a key reaches a find() that is not
    // inlined, the load waits for the stores to retire and the lookups stop overlapping.
    // Updating 300k random cells through a std::unordered_map took 2.3 times as long.
    //
    // This replaced OpenVDB's `x * 73856093 ^ y * 19349669 ^ z * 83492791`, which collided
    // heavily on any grid crossing zero once the coordinates were sign extended (the 46656
    // root keys of a 36^3 grid centred on the origin gave 29226 distinct values), was
    // truncated to 20 bits on top of that, and left its low bits at zero when the
    // coordinates shared theirs, as root keys do.
    uint64_t k = (uint64_t(uint32_t(p.x)) * UINT64_C(0x9e3779b97f4a7c15)) ^
                 (uint64_t(uint32_t(p.y)) * UINT64_C(0xc2b2ae3d27d4eb4f)) ^
                 (uint64_t(uint32_t(p.z)) * UINT64_C(0x165667b19e3779f9));
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return static_cast<std::size_t>(k);
  }
};

}  // namespace std

#ifndef BONXAI_NOINLINE
#if defined(__GNUC__) || defined(__clang__)
#define BONXAI_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define BONXAI_NOINLINE __declspec(noinline)
#else
#define BONXAI_NOINLINE
#endif
#endif

namespace Bonxai {

/**
 * @brief A hash map from CoordT to ValueT: the root map of a VoxelGrid.
 *
 * It has the interface of a std::unordered_map, reduced to what Bonxai uses, and is
 * built differently:
 *
 * - Every value is allocated on its own and never moves, whatever is inserted or erased
 *   afterwards: the accessors of a VoxelGrid cache pointers to them.
 * - The index is open addressing with linear probing over buckets of 16 bytes, holding the
 *   upper half of the hash, the position of the value and a pointer to it. A lookup goes
 *   from its bucket straight to the value, where std::unordered_map reads two other nodes
 *   first; a key that is not there is almost always rejected without leaving the buckets;
 *   and growing the index moves buckets only, never values.
 * - It is iterated in insertion order, which is also the order in which the values were
 *   allocated: a traversal walks the memory forward. Erasing a key moves the last value
 *   into its place.
 *
 * Its hash is internal and only good in its upper bits, which are the only ones it uses:
 * std::hash<CoordT> is the one to use in other containers.
 *
 * Inserting or erasing invalidates the iterators. References and pointers to a value stay
 * valid until that very value is erased.
 */
template <typename ValueT>
class CoordMap {
 public:
  using key_type = CoordT;
  using mapped_type = ValueT;
  using value_type = std::pair<const CoordT, ValueT>;
  using size_type = std::size_t;

  template <bool IS_CONST>
  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = CoordMap::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = std::conditional_t<IS_CONST, const value_type&, value_type&>;
    using pointer = std::conditional_t<IS_CONST, const value_type*, value_type*>;

    Iterator() = default;

    /// an iterator converts to a const_iterator
    template <bool OTHER, typename = std::enable_if_t<IS_CONST && !OTHER>>
    Iterator(const Iterator<OTHER>& other)
        : values_(other.values_), pos_(other.pos_), value_(other.value_) {}

    reference operator*() const { return *value_; }
    pointer operator->() const { return value_; }

    Iterator& operator++() {
      ++pos_;
      value_ = pos_ < values_->size() ? (*values_)[pos_] : nullptr;
      return *this;
    }
    Iterator operator++(int) {
      Iterator previous = *this;
      ++(*this);
      return previous;
    }

    // every position holds a distinct value, and end() holds none
    friend bool operator==(const Iterator& a, const Iterator& b) { return a.value_ == b.value_; }
    friend bool operator!=(const Iterator& a, const Iterator& b) { return a.value_ != b.value_; }

   private:
    friend class CoordMap;
    template <bool>
    friend class Iterator;

    // the value is carried along with its position, so that dereferencing the result of
    // find() does not read the vector of values again
    Iterator(const std::vector<value_type*>* values, size_t pos, value_type* value)
        : values_(values), pos_(pos), value_(value) {}

    const std::vector<value_type*>* values_ = nullptr;
    size_t pos_ = 0;
    value_type* value_ = nullptr;
  };

  using iterator = Iterator<false>;
  using const_iterator = Iterator<true>;

  CoordMap() = default;

  CoordMap(const CoordMap&) = delete;
  CoordMap& operator=(const CoordMap&) = delete;

  CoordMap(CoordMap&& other) noexcept { swap(other); }
  CoordMap& operator=(CoordMap&& other) noexcept {
    CoordMap(std::move(other)).swap(*this);
    return *this;
  }

  ~CoordMap() { clear(); }

  void swap(CoordMap& other) noexcept {
    std::swap(values_, other.values_);
    std::swap(buckets_, other.buckets_);
    std::swap(mask_, other.mask_);
    std::swap(shift_, other.shift_);
    std::swap(bits_, other.bits_);
  }

  [[nodiscard]] iterator begin() { return iteratorAt(0); }
  [[nodiscard]] iterator end() { return {&values_, values_.size(), nullptr}; }
  [[nodiscard]] const_iterator begin() const { return const_cast<CoordMap*>(this)->begin(); }
  [[nodiscard]] const_iterator end() const { return const_cast<CoordMap*>(this)->end(); }
  [[nodiscard]] const_iterator cbegin() const { return begin(); }
  [[nodiscard]] const_iterator cend() const { return end(); }

  [[nodiscard]] size_type size() const { return values_.size(); }
  [[nodiscard]] bool empty() const { return values_.empty(); }

  [[nodiscard]] iterator find(const CoordT& key) {
    const Bucket& bucket = buckets_[findBucket(key, tagOf(key))];
    return {&values_, bucket.pos, bucket.value};
  }
  [[nodiscard]] const_iterator find(const CoordT& key) const {
    return const_cast<CoordMap*>(this)->find(key);
  }

  /// Inserts {key, ValueT(args...)} if the key is missing, and does nothing otherwise.
  template <typename... Args>
  std::pair<iterator, bool> try_emplace(const CoordT& key, Args&&... args) {
    const uint32_t tag = tagOf(key);
    const Bucket& bucket = buckets_[findBucket(key, tag)];
    if (bucket.value) {
      return {iterator(&values_, bucket.pos, bucket.value), false};
    }
    return {emplaceNew(key, tag, std::forward<Args>(args)...), true};
  }

  std::pair<iterator, bool> insert(value_type&& value) {
    return try_emplace(value.first, std::move(value.second));
  }

  size_type erase(const CoordT& key) {
    size_t hole = findBucket(key, tagOf(key));
    value_type* erased = buckets_[hole].value;
    if (!erased) {
      return 0;
    }
    const uint32_t pos = buckets_[hole].pos;

    // Backward shift deletion: pull back the buckets that follow in the same cluster,
    // so that the hole does not cut the probe sequence of any of them.
    const size_t mask = mask_;
    for (size_t i = (hole + 1) & mask; buckets_[i].value; i = (i + 1) & mask) {
      const size_t home = homeOf(buckets_[i].tag);
      // bucket i can move to the hole, unless its home is in (hole, i]
      if (((i - home) & mask) >= ((i - hole) & mask)) {
        buckets_[hole] = buckets_[i];
        hole = i;
      }
    }
    buckets_[hole] = Bucket{};

    // fill the gap in the vector with the last value, and update its bucket
    value_type* last = values_.back();
    if (last != erased) {
      values_[pos] = last;
      for (size_t i = homeOf(tagOf(last->first));; i = (i + 1) & mask) {
        if (buckets_[i].value == last) {
          buckets_[i].pos = pos;
          break;
        }
      }
    }
    values_.pop_back();
    delete erased;
    return 1;
  }

  /// Destroys all the values, and releases all the memory.
  void clear() {
    // Last to first. Front to back frees the memory in increasing addresses: glibc keeps
    // the first few blocks of each size in its thread cache and merges all the others into
    // the top of the heap, which then goes back to the kernel, to be faulted in again by
    // the next grid. Rebuilding a small grid in a loop took 3.5 times as long.
    // (ankerl::unordered_dense's segmented_map had the same issue, fixed in 5.1.0.)
    for (auto it = values_.rbegin(); it != values_.rend(); ++it) {
      delete *it;
    }
    // not `values_ = {}`: assigning an initializer list keeps the capacity
    std::vector<value_type*>().swap(values_);
    setBuckets(emptyBuckets(), 0);
  }

  void reserve(size_type count) {
    uint32_t bits = std::max(bits_, kMinBits);
    while (bits <= kMaxBits && (size_t(1) << bits) / 2 < count) {
      ++bits;
    }
    if (bits != bits_) {
      rehash(bits);
    }
    values_.reserve(count);
  }

  /// Bytes used by the index: the buckets, and the vector of pointers to the values.
  /// The values themselves are not counted.
  [[nodiscard]] size_t memUsage() const {
    const size_t buckets = bits_ == 0 ? 0 : mask_ + 1;
    return buckets * sizeof(Bucket) + values_.capacity() * sizeof(value_type*);
  }

 private:
  struct Bucket {
    uint32_t tag = 0;             // the upper 32 bits of the hash
    uint32_t pos = 0;             // position of the value in values_
    value_type* value = nullptr;  // nullptr: the bucket is empty
  };

  static constexpr uint32_t kMinBits = 4;
  // positions are 32 bits, and a size_t of 32 bits cannot be shifted by 32
  static constexpr uint32_t kMaxBits =
      std::min<uint32_t>(32, std::numeric_limits<size_t>::digits - 1);

  std::vector<value_type*> values_;  // in insertion order
  // Until the first insertion, the map points at a single, static, empty bucket, rather
  // than at nullptr: a lookup then needs no special case. Lookups are latency bound, and
  // every instruction added to their path lets the CPU overlap fewer of them.
  // bits_ == 0 tells that bucket apart; its address is never compared, since a copy of
  // the static may exist in each shared library.
  Bucket* buckets_ = emptyBuckets();
  size_t mask_ = 0;      // number of buckets - 1
  uint32_t shift_ = 32;  // the home bucket of a tag is tag >> shift_
  uint32_t bits_ = 0;    // log2 of the number of buckets, 0 before the first insertion

  static Bucket* emptyBuckets() {
    static Bucket empty[1];
    return empty;
  }

  /// One multiplication of the packed coordinates: its upper bits, the only ones used
  /// here, depend on every bit of x, y and z.
  static uint32_t tagOf(const CoordT& key) {
    const uint64_t packed = (uint64_t(uint32_t(key.x)) | (uint64_t(uint32_t(key.y)) << 32)) ^
                            (uint64_t(uint32_t(key.z)) * UINT64_C(0x9e3779b97f4a7c15));
    return uint32_t((packed * UINT64_C(0xd6e8feb86659fd93)) >> 32);
  }

  /// at most half full: a hit reads 1.5 buckets on average, a miss 2.5
  [[nodiscard]] size_t maxLoad() const { return bits_ == 0 ? 0 : (mask_ + 1) / 2; }
  [[nodiscard]] size_t homeOf(uint32_t tag) const {
    // 64 bits, so that the shift by 32 of the empty map is defined
    return size_t(uint64_t(tag) >> shift_);
  }

  /// the bucket holding the key, or the empty bucket where its probe sequence ends
  [[nodiscard]] size_t findBucket(const CoordT& key, uint32_t tag) const {
    for (size_t i = homeOf(tag);; i = (i + 1) & mask_) {
      const Bucket& bucket = buckets_[i];
      if (!bucket.value || (bucket.tag == tag && bucket.value->first == key)) {
        return i;
      }
    }
  }

  void setBuckets(Bucket* buckets, uint32_t bits) {
    if (bits_ != 0) {
      delete[] buckets_;
    }
    buckets_ = buckets;
    bits_ = bits;
    mask_ = (size_t(1) << bits) - 1;
    shift_ = 32 - bits;
  }

  // The cold path of try_emplace, and never inlined into it. Inlined, the address of the
  // key escapes into the construction of the pair, and the compiler keeps the caller's
  // key in memory, written as three 32 bits stores. VoxelGrid's accessors then copy that
  // key into their cache with a 64 bits load, which cannot be forwarded from those stores
  // and has to wait for them to retire: lookups that should overlap run one after the
  // other, and updating a large grid got 2.5 times slower. Taken by value, the key reaches
  // this function in registers.
  template <typename... Args>
  BONXAI_NOINLINE iterator emplaceNew(CoordT key, uint32_t tag, Args&&... args) {
    // make room first: from here on, nothing can throw once the value exists
    if (values_.size() + 1 > maxLoad()) {
      rehash(std::max(bits_ + 1, kMinBits));
    }
    if (values_.size() == values_.capacity()) {
      values_.reserve(values_.empty() ? size_t(1) << kMinBits : 2 * values_.capacity());
    }
    auto* value = new value_type(std::piecewise_construct, std::forward_as_tuple(key),
                                 std::forward_as_tuple(std::forward<Args>(args)...));
    const uint32_t pos = uint32_t(values_.size());
    values_.push_back(value);
    buckets_[findBucket(key, tag)] = {tag, pos, value};
    return iterator(&values_, pos, value);
  }

  void rehash(uint32_t bits) {
    if (bits > kMaxBits) {
      throw std::length_error("CoordMap: too many elements");
    }
    Bucket* buckets = new Bucket[size_t(1) << bits];
    const size_t mask = (size_t(1) << bits) - 1;
    for (size_t i = 0; i <= mask_; ++i) {
      if (buckets_[i].value) {
        size_t b = buckets_[i].tag >> (32 - bits);
        while (buckets[b].value) {
          b = (b + 1) & mask;
        }
        buckets[b] = buckets_[i];
      }
    }
    setBuckets(buckets, bits);
  }

  [[nodiscard]] iterator iteratorAt(size_t pos) {
    return {&values_, pos, pos < values_.size() ? values_[pos] : nullptr};
  }
};

}  // namespace Bonxai

// ---- grid_allocator.hpp ----

#include <cassert>
#include <memory>
#include <mutex>
#include <stdexcept>

// ---- mask.hpp ----

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace Bonxai {

class Mask {
  uint64_t* words_ = nullptr;
  // small object optimization, that will be used when
  // SIZE <= 512 bits, i.e LOG2DIM <= 3
  uint64_t static_words_[8];

 public:
  /// Initialize all bits to zero.
  Mask(size_t log2dim);
  /// Initialize all bits to a given value.
  Mask(size_t log2dim, bool on);

  Mask(const Mask& other);
  Mask(Mask&& other);

  /// Declaring the move constructor suppressed the implicit assignment operators;
  /// without this one, Grid::operator=(Grid&&) does not compile when instantiated.
  Mask& operator=(Mask&& other);

  ~Mask();

  /// Return the memory footprint in bytes of this Mask
  size_t memUsage() const;

  /// Return the number of bits available in this Mask
  uint32_t bitCount() const { return SIZE; }

  /// Return the number of machine words used by this Mask
  uint32_t wordCount() const { return WORD_COUNT; }

  uint64_t getWord(size_t n) const { return words_[n]; }

  void setWord(size_t n, uint64_t v) {
    words_[n] = (n + 1 == WORD_COUNT) ? (v & lastWordMask()) : v;
  }

  uint32_t countOn() const;

  /// Iterates the ON bits, keeping the rest of the current word in a register
  /// and popping them with `w &= w - 1`.
  class Iterator {
   public:
    Iterator(const Mask* parent)
        : pos_(parent->SIZE), word_(0), word_index_(parent->WORD_COUNT), parent_(parent) {}

    Iterator(uint32_t pos, const Mask* parent) : Iterator(parent) {
      if (pos < parent->SIZE) {
        pos_ = pos;
        word_index_ = pos >> 6;
        // bits still to visit: strictly above `pos`
        word_ = parent->words_[word_index_] & (~uint64_t(1) << (pos & 63u));
      }
    }

    Iterator& operator=(const Iterator&) = default;

    uint32_t operator*() const { return pos_; }

    operator bool() const { return pos_ != parent_->SIZE; }

    Iterator& operator++() {
      while (word_ == 0) {
        if (++word_index_ >= parent_->WORD_COUNT) {
          pos_ = parent_->SIZE;
          return *this;
        }
        word_ = parent_->words_[word_index_];
      }
      pos_ = (word_index_ << 6) + FindLowestOn(word_);
      word_ &= word_ - 1;
      return *this;
    }

   private:
    uint32_t pos_;
    uint64_t word_;
    uint32_t word_index_;
    const Mask* parent_;
  };

  bool operator==(const Mask& other) const;

  bool operator!=(const Mask& other) const { return !((*this) == other); }

  Iterator beginOn() const { return Iterator(this->findFirstOn(), this); }

  /// Return true if the given bit is set.
  bool isOn(uint32_t n) const;
  /// Return true if any bit is set.
  bool isOn() const;

  bool isOff() const;

  bool setOn(uint32_t n);

  bool setOff(uint32_t n);

  void set(uint32_t n, bool On);

  /// Set all bits on
  void setOn();

  /// Set all bits off
  void setOff();

  /// Set all bits to the value "on"
  void set(bool on);

  /// Toggle the state of all bits in the mask
  void toggle();
  /// Toggle the state of one bit in the mask
  void toggle(uint32_t n);

  uint32_t findFirstOn() const;

  uint32_t size() const { return SIZE; }

 private:
  /// bits of the last word that are inside SIZE
  uint64_t lastWordMask() const;

  static uint32_t FindLowestOn(uint64_t v);
  static uint32_t CountOn(uint64_t v);

 public:
  // Legacy metadata; callers should treat these values as read-only.
  // Number of bits in mask
  uint32_t SIZE;
  // Number of 64 bit words
  uint32_t WORD_COUNT;
};

//----------------------------------------------------
//----------------- Implementations ------------------
//----------------------------------------------------

#define BONXAI_USE_INTRINSICS

/// Returns the index of the lowest, i.e. least significant, on bit in
/// the specified 64 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v !=
/// uint32_t(0)!

inline uint32_t Mask::FindLowestOn(uint64_t v) {
#if defined(_MSC_VER) && defined(BONXAI_USE_INTRINSICS)
  unsigned long index;
  _BitScanForward64(&index, v);
  return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(BONXAI_USE_INTRINSICS)
  return static_cast<uint32_t>(__builtin_ctzll(v));
#else
  static const unsigned char DeBruijn[64] = {
      0,  1,  2,  53, 3,  7,  54, 27, 4,  38, 41, 8,  34, 55, 48, 28, 62, 5,  39, 46, 44, 42,
      22, 9,  24, 35, 59, 56, 49, 18, 29, 11, 63, 52, 6,  26, 37, 40, 33, 47, 61, 45, 43, 21,
      23, 58, 17, 10, 51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
  };
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
  /// @warning evil bit twiddling ahead!
  return DeBruijn[uint64_t((v & -v) * UINT64_C(0x022FDD63CC95386D)) >> 58];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}

/// @return Number of bits that are on in the specified 64-bit word

inline uint32_t Mask::CountOn(uint64_t v) {
#if defined(_MSC_VER) && defined(_M_X64)
  v = __popcnt64(v);
#elif (defined(__GNUC__) || defined(__clang__))
  v = __builtin_popcountll(v);
#else
  // Software Implementation
  /// @warning evil bit twiddling ahead!
  v = v - ((v >> 1) & uint64_t(0x5555555555555555));
  v = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
  v = (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) * uint64_t(0x101010101010101)) >> 56;
#endif
  return static_cast<uint32_t>(v);
}

inline uint64_t Mask::lastWordMask() const {
  // SIZE can be smaller than a word (log2dim == 1 gives 8 bits): the padding
  // must stay off, or countOn() and isOn() would report bits that do not exist
  const uint32_t bits = SIZE & 63u;
  return bits == 0 ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1);
}

inline void Mask::setOn() { this->set(true); }

inline void Mask::setOff() { this->set(false); }

inline void Mask::set(bool on) {
  const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
  for (uint32_t i = 0; i < WORD_COUNT; ++i) {
    words_[i] = v;
  }
  words_[WORD_COUNT - 1] &= lastWordMask();
}

inline void Mask::toggle() {
  uint32_t n = WORD_COUNT;
  for (auto* w = words_; n--; ++w) {
    *w = ~*w;
  }
  words_[WORD_COUNT - 1] &= lastWordMask();
}

inline void Mask::toggle(uint32_t n) { words_[n >> 6] ^= uint64_t(1) << (n & 63); }

inline uint32_t Mask::findFirstOn() const {
  const uint64_t* w = words_;
  uint32_t n = 0;
  while (n < WORD_COUNT && !*w) {
    ++w;
    ++n;
  }
  return n == WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(*w);
}

inline Mask::Mask(size_t log2dim) : Mask(log2dim, false) {}

inline Mask::Mask(size_t log2dim, bool on)
    : SIZE(1U << (3 * log2dim)), WORD_COUNT(std::max(SIZE >> 6, 1u)) {
  words_ = (WORD_COUNT <= 8) ? static_words_ : new uint64_t[WORD_COUNT];
  this->set(on);
}

inline Mask::Mask(const Mask& other) : SIZE(other.SIZE), WORD_COUNT(other.WORD_COUNT) {
  words_ = (WORD_COUNT <= 8) ? static_words_ : new uint64_t[WORD_COUNT];

  for (uint32_t i = 0; i < WORD_COUNT; ++i) {
    words_[i] = other.words_[i];
  }
}

inline Mask::Mask(Mask&& other) : SIZE(other.SIZE), WORD_COUNT(other.WORD_COUNT) {
  if (WORD_COUNT <= 8) {
    words_ = static_words_;
    for (uint32_t i = 0; i < WORD_COUNT; ++i) {
      words_[i] = other.words_[i];
    }
  } else {
    std::swap(words_, other.words_);
  }
}

inline Mask& Mask::operator=(Mask&& other) {
  if (this == &other) {
    return *this;
  }
  if (WORD_COUNT > 8) {
    delete[] words_;
  }
  SIZE = other.SIZE;
  WORD_COUNT = other.WORD_COUNT;

  if (WORD_COUNT <= 8) {
    // the words of the other Mask live inside the other Mask: they must be copied
    words_ = static_words_;
    for (uint32_t i = 0; i < WORD_COUNT; ++i) {
      words_[i] = other.words_[i];
    }
  } else {
    words_ = other.words_;
    other.words_ = nullptr;
  }
  return *this;
}

inline Mask::~Mask() {
  if (WORD_COUNT > 8) {
    delete[] words_;
  }
}

inline size_t Mask::memUsage() const {
  if (WORD_COUNT > 8) {
    return sizeof(Mask) + sizeof(uint64_t) * WORD_COUNT;
  }
  return sizeof(Mask);
}

inline uint32_t Mask::countOn() const {
  uint32_t sum = 0, n = WORD_COUNT;
  for (const uint64_t* w = words_; n--; ++w) {
    sum += CountOn(*w);
  }
  return sum;
}

inline bool Mask::operator==(const Mask& other) const {
  if (SIZE != other.SIZE) {
    return false;
  }
  for (uint32_t i = 0; i < WORD_COUNT; ++i) {
    if (words_[i] != other.words_[i]) {
      return false;
    }
  }
  return true;
}

inline bool Mask::isOn(uint32_t n) const {
  return 0 != (words_[n >> 6] & (uint64_t(1) << (n & 63)));
}

inline bool Mask::isOn() const {
  for (uint32_t i = 0; i + 1 < WORD_COUNT; ++i) {
    if (words_[i] != ~uint64_t(0)) {
      return false;
    }
  }
  return words_[WORD_COUNT - 1] == lastWordMask();
}

inline bool Mask::isOff() const {
  for (uint32_t i = 0; i < WORD_COUNT; ++i) {
    if (words_[i] != uint64_t(0)) {
      return false;
    }
  }
  return true;
}

inline bool Mask::setOn(uint32_t n) {
  uint64_t& word = words_[n >> 6];
  const uint64_t on_bit = (uint64_t(1) << (n & 63));
  bool was_on = word & on_bit;
  word |= on_bit;
  return was_on;
}

inline bool Mask::setOff(uint32_t n) {
  uint64_t& word = words_[n >> 6];
  const uint64_t on_bit = (uint64_t(1) << (n & 63));
  bool was_on = word & on_bit;
  word &= ~(on_bit);
  return was_on;
}

inline void Mask::set(uint32_t n, bool On) {
#if 1  // switch between branchless
  auto& word = words_[n >> 6];
  n &= 63;
  word &= ~(uint64_t(1) << n);
  word |= uint64_t(On) << n;
#else
  On ? this->setOn(n) : this->setOff(n);
#endif
}

}  // namespace Bonxai

namespace Bonxai {

/**
 * @brief The GridBlockAllocator is used to pre-allocate the meory of multiple Grids
 * in "chunks". It is a very simple memory pool.
 *
 * Each chunk allocates memory for 512 Grids
 */
template <typename DataT>
class GridBlockAllocator {
 public:
  // use log2dim of the block to be allocated
  GridBlockAllocator(size_t log2dim);

  GridBlockAllocator(const GridBlockAllocator&) = delete;
  GridBlockAllocator(GridBlockAllocator&&) = default;

  GridBlockAllocator& operator=(const GridBlockAllocator& other) = delete;
  GridBlockAllocator& operator=(GridBlockAllocator&& other) = default;

  /// What the leaves need when they give their block back. It does not live in the
  /// allocator, which moves with its VoxelGrid: every chunk points to it, and so every
  /// leaf, through the chunk its Deleter holds.
  struct State {
    std::mutex mutex;
    size_t size = 0;  // blocks in use
  };

  struct Chunk {
    explicit Chunk(std::shared_ptr<State> s) : mask(3, true), state(std::move(s)) {}
    Mask mask;
    std::shared_ptr<State> state;
    // not a std::vector: resize() would memset the whole chunk, and no cell is
    // read before it is written
    std::unique_ptr<char[]> data;
  };

  /// Gives a block back to the pool. A concrete type rather than a
  /// std::function, whose capture would not fit the small-object buffer and so
  /// cost one heap allocation per leaf.
  class Deleter {
   public:
    Deleter() = default;
    Deleter(std::shared_ptr<Chunk> chunk, uint32_t index)
        : chunk_(std::move(chunk)), index_(index) {}

    void operator()() const {
      assert(index_ < blocks_per_chunk);
      State& state = *chunk_->state;
      std::unique_lock lock(state.mutex);
      chunk_->mask.setOn(index_);
      state.size--;
    }

   private:
    std::shared_ptr<Chunk> chunk_;
    uint32_t index_ = 0;
  };

  std::pair<DataT*, Deleter> allocateBlock();

  /// Leaves still alive give their blocks back to the chunks they came from, which they
  /// keep alive, and not to the new ones.
  void clear() {
    chunks_.clear();
    state_ = std::make_shared<State>();
    capacity_ = 0;
  }

  void releaseUnusedMemory();

  size_t capacity() const { return capacity_; }

  size_t size() const { return state_ ? state_->size : 0; }

  size_t memUsage() const;

  // specific size to use for Mask(3)
  static constexpr size_t blocks_per_chunk = 512;

 protected:
  size_t log2dim_ = 0;
  size_t block_bytes_ = 0;
  size_t capacity_ = 0;
  std::vector<std::shared_ptr<Chunk>> chunks_;
  std::shared_ptr<State> state_;

  void addNewChunk();
};

//----------------------------------------------------
//----------------- Implementations ------------------
//----------------------------------------------------

template <typename DataT>
inline GridBlockAllocator<DataT>::GridBlockAllocator(size_t log2dim)
    : log2dim_(log2dim),
      block_bytes_(std::pow((1 << log2dim), 3) * sizeof(DataT)),
      state_(std::make_shared<State>()) {}

template <typename DataT>
inline std::pair<DataT*, typename GridBlockAllocator<DataT>::Deleter>
GridBlockAllocator<DataT>::allocateBlock() {
  // A moved-from allocator has no chunks or state, but remains reusable.
  if (!state_) {
    clear();
  }
  std::unique_lock lock(state_->mutex);
  if (state_->size >= capacity_) {
    // Need more memory. Create a new chunk
    addNewChunk();
    // first index of new chunk is available
    std::shared_ptr<Chunk> chunk = chunks_.back();
    DataT* ptr = reinterpret_cast<DataT*>(chunk->data.get());
    chunk->mask.setOff(0);
    state_->size++;
    return {ptr, Deleter(chunk, 0)};
  }

  // There must be available memory, somewhere. Search in reverse order
  for (auto it = chunks_.rbegin(); it != chunks_.rend(); it++) {
    std::shared_ptr<Chunk>& chunk = (*it);
    auto mask_index = chunk->mask.findFirstOn();
    if (mask_index < chunk->mask.size()) {
      // found in this chunk
      size_t data_index = block_bytes_ * mask_index;
      DataT* ptr = reinterpret_cast<DataT*>(&chunk->data[data_index]);
      chunk->mask.setOff(mask_index);
      state_->size++;
      return {ptr, Deleter(chunk, mask_index)};
    }
  }
  throw std::logic_error("Unexpected end of GridBlockAllocator::allocateBlock");
}

template <typename DataT>
inline void GridBlockAllocator<DataT>::releaseUnusedMemory() {
  if (!state_) {
    return;
  }
  std::unique_lock lock(state_->mutex);
  int to_be_erased_count = 0;
  auto remove_if = std::remove_if(chunks_.begin(), chunks_.end(), [&](const auto& chunk) -> bool {
    bool notUsed = chunk->mask.isOn();
    to_be_erased_count += (notUsed) ? 1 : 0;
    return notUsed;
  });
  chunks_.erase(remove_if, chunks_.end());
  capacity_ -= to_be_erased_count * blocks_per_chunk;
}

template <typename DataT>
inline size_t GridBlockAllocator<DataT>::memUsage() const {
  return chunks_.size() * (sizeof(Chunk) + block_bytes_ * blocks_per_chunk);
}

template <typename DataT>
inline void GridBlockAllocator<DataT>::addNewChunk() {
  auto chunk = std::make_shared<Chunk>(state_);
  chunk->data.reset(new char[blocks_per_chunk * block_bytes_]);
  chunks_.push_back(chunk);
  capacity_ += blocks_per_chunk;
}

}  // namespace Bonxai

namespace Bonxai {

// empty data structure used exclusively in BinaryVoxelGrid
struct EmptyVoxel {};

/**
 * @brief The Grid class is used to store data in a cube.
 * The size (DIM) of the cube can only be a power of 2.
 *
 * For instance, given Grid(3),
 * DIM will be 8 and SIZE 512 (8³)
 */
template <typename DataT>
class Grid {
 private:
  uint8_t dim_ = 0;
  // total number of elements in the cube
  uint32_t size_ = 0;

  DataT* data_ = nullptr;
  Bonxai::Mask mask_;
  bool external_memory_ = false;

 public:
  Grid(size_t log2dim) : dim_(1 << log2dim), mask_(log2dim) {
    if constexpr (!std::is_same_v<DataT, EmptyVoxel>) {
      size_ = dim_ * dim_ * dim_;
      data_ = new DataT[size_];
    }
  }

  Grid(size_t log2dim, DataT* preAllocatedMemory)
      : dim_(1 << log2dim),
        size_(dim_ * dim_ * dim_),
        data_(preAllocatedMemory),
        mask_(log2dim),
        external_memory_(true) {}

  Grid(const Grid& other) = delete;
  Grid& operator=(const Grid& other) = delete;

  Grid(Grid&& other);
  Grid& operator=(Grid&& other);

  ~Grid() {
    if (data_ != nullptr && !external_memory_) {
      delete[] data_;
    }
  }

  [[nodiscard]] size_t memUsage() const;

  [[nodiscard]] size_t size() const { return size_; }

  [[nodiscard]] Bonxai::Mask& mask() { return mask_; };

  [[nodiscard]] const Bonxai::Mask& mask() const { return mask_; }

  [[nodiscard]] DataT& cell(size_t index) {
    static_assert(!std::is_same_v<DataT, EmptyVoxel>,
                  "Cells have no value, when DataT == EmptyVoxel");
    return data_[index];
  }

  [[nodiscard]] const DataT& cell(size_t index) const {
    static_assert(!std::is_same_v<DataT, EmptyVoxel>,
                  "Cells have no value, when DataT == EmptyVoxel");
    return data_[index];
  }
};

//----------------------------------------------------------

/// Branching factors as a policy. StaticShape lets the index arithmetic fold
/// into constant shifts; DynamicShape is for shapes only known at runtime,
/// such as a grid read back by Deserialize().
template <uint32_t INNER = 2, uint32_t LEAF = 3>
struct StaticShape {
  static_assert(INNER >= 1, "The minimum value of inner_bits is 1");
  static_assert(LEAF >= 1, "The minimum value of leaf_bits is 1");

  static constexpr uint32_t INNER_BITS = INNER;
  static constexpr uint32_t LEAF_BITS = LEAF;
  static constexpr uint32_t LOG2_N = INNER_BITS + LEAF_BITS;
  static constexpr uint32_t Log2N = LOG2_N;
  static constexpr uint32_t INNER_MASK = (uint32_t(1) << INNER_BITS) - 1;
  static constexpr uint32_t LEAF_MASK = (uint32_t(1) << LEAF_BITS) - 1;

  StaticShape() = default;

  /// Lets VoxelGrid keep one constructor; rejects a shape it cannot represent.
  StaticShape(uint32_t inner_bits, uint32_t leaf_bits) {
    if (inner_bits != INNER_BITS || leaf_bits != LEAF_BITS) {
      throw std::runtime_error(
          "This VoxelGrid has a compile-time shape that does not match the requested "
          "inner_bits / leaf_bits. Use VoxelGrid<DataT, DynamicShape> instead.");
    }
  }
};

struct DynamicShape {
  uint32_t INNER_BITS = 2;
  uint32_t LEAF_BITS = 3;
  uint32_t LOG2_N = INNER_BITS + LEAF_BITS;
  uint32_t Log2N = LOG2_N;
  uint32_t INNER_MASK = (uint32_t(1) << INNER_BITS) - 1;
  uint32_t LEAF_MASK = (uint32_t(1) << LEAF_BITS) - 1;

  DynamicShape() = default;

  DynamicShape(uint32_t inner_bits, uint32_t leaf_bits)
      : INNER_BITS(inner_bits),
        LEAF_BITS(leaf_bits),
        LOG2_N(inner_bits + leaf_bits),
        Log2N(LOG2_N),
        INNER_MASK((uint32_t(1) << inner_bits) - 1),
        LEAF_MASK((uint32_t(1) << leaf_bits) - 1) {
    if (INNER_BITS < 1 || LEAF_BITS < 1) {
      throw std::runtime_error("The minimum value of the inner_bits and leaf_bits should be 1");
    }
  }
};

//----------------------------------------------------------

enum ClearOption {
  // reset the entire grid, freeing all the memory allocated so far
  CLEAR_MEMORY,
  // keep the memory allocated, but set all the cells to OFF
  SET_ALL_CELLS_OFF
};

template <typename DataT, typename Shape = DynamicShape>
class VoxelGrid : public Shape {
 public:
  // Legacy public metadata; callers should treat these values as read-only.
  double resolution;
  double inv_resolution;

 private:
  GridBlockAllocator<DataT> leaf_block_allocator_;

  // Bumped whenever inner/leaf nodes are released, so that the Accessors can tell that the
  // pointers they cached do not refer to live nodes anymore. Moving a grid bumps it too, on
  // both sides: the moved-from grid's accessors cached nodes it no longer owns, and a
  // move-assigned grid's accessors cached nodes that were destroyed.
  struct CacheEpoch {
    uint32_t value = 0;

    CacheEpoch() = default;
    CacheEpoch(CacheEpoch&& other) noexcept : value(other.value) { ++other.value; }
    CacheEpoch& operator=(CacheEpoch&& other) noexcept {
      ++value;
      ++other.value;
      return *this;
    }
  };
  CacheEpoch cache_epoch_;

 public:
  using LeafGrid = Grid<DataT>;
  using InnerGrid = Grid<std::shared_ptr<LeafGrid>>;
  using RootMap = CoordMap<InnerGrid>;

  /**
   * @brief VoxelGrid constructor
   *
   * @param voxel_size  dimension of the voxel. Used to convert between Point3D and
   * CoordT
   */
  VoxelGrid(double voxel_size, uint8_t inner_bits = Shape{}.INNER_BITS,
            uint8_t leaf_bits = Shape{}.LEAF_BITS);

  VoxelGrid(const VoxelGrid&) = delete;
  VoxelGrid& operator=(const VoxelGrid&) = delete;

  VoxelGrid(VoxelGrid&& other) = default;
  VoxelGrid& operator=(VoxelGrid&& other) = default;

  uint32_t innetBits() const { return this->INNER_BITS; }
  uint32_t leafBits() const { return this->LEAF_BITS; }
  double voxelSize() const { return resolution; }

  const RootMap& rootMap() const { return root_map; }
  /// Erasing a root through this, or clearing the map, frees nodes that accessors may have
  /// cached: create the accessors again afterwards. clear() and releaseUnusedMemory() do
  /// not have this problem.
  RootMap& rootMap() { return root_map; }

  /// @brief getMemoryUsage returns the amount of bytes used by this data structure
  [[nodiscard]] size_t memUsage() const;

  /**
   *  Try freeing memory;  this will discard grids where all the cells are OFF.
   *  Note that the memory release is NOT guaranteed, since we are using a memory pool too.
   *  Note that this invalidates the cache of the existing Accessors, but they will detect it
   *  and refresh themselves transparently on the next access.
   */
  void releaseUnusedMemory();

  /// @brief Return the total number of active cells
  [[nodiscard]] size_t activeCellsCount() const;

  /// @brief posToCoord is used to convert real coordinates to CoordT indices.
  [[nodiscard]] CoordT posToCoord(double x, double y, double z) const;

  /// @brief posToCoord is used to convert real coordinates to CoordT indices.
  [[nodiscard]] CoordT posToCoord(const Point3D& pos) const {
    return posToCoord(pos.x, pos.y, pos.z);
  }

  /// @brief coordToPos converts CoordT indices to Point3D.
  [[nodiscard]] Point3D coordToPos(const CoordT& coord) const;

  /**
   *  @brief forEachCell apply a function of type:
   *
   *      void(const DataT&, const CoordT&)
   *
   * to each active element of the grid.
   */
  template <class VisitorFunction>
  void forEachCell(VisitorFunction func) const;

  /**
   *  @brief forEachCell apply a function of type:
   *
   *      void(DataT&, const CoordT&)
   *
   * to each active element of the grid.
   */
  template <class VisitorFunction>
  void forEachCell(VisitorFunction func) {
    static_cast<const VoxelGrid*>(this)->forEachCell(func);
  }

  void clear(ClearOption opt = CLEAR_MEMORY);

  // Original spelling retained for existing callers.
  void Clear() { clear(CLEAR_MEMORY); }

  // You should never use this function directly. It is exposed in the public API only
  // for special cases, such as serialization or testing
  std::shared_ptr<LeafGrid> allocateLeafGrid();

  class ConstAccessor {
   public:
    ConstAccessor(const VoxelGrid& grid) : grid_(grid), cache_epoch_(grid.cache_epoch_.value) {}

    /** @brief value getter.
     *
     * @param coord   coordinate of the cell.
     * @return        return the pointer to the value or nullptr if not set.
     */
    [[nodiscard]] const DataT* value(const CoordT& coord) const;

    /**
     * @brief isCellOn only check if a cell is in "On" state
     * @param coordinate of the cell.
     */
    [[nodiscard]] bool isCellOn(const CoordT& coord) const;

    /// @brief lastInnerGrid returns the pointer to the InnerGrid in the cache.
    [[nodiscard]] const InnerGrid* lastInnerGrid() const {
      refreshCache();
      return prev_inner_ptr_;
    }

    // Original API contained this spelling; retain it as an alias.
    [[nodiscard]] const InnerGrid* lastInnerdGrid() const { return lastInnerGrid(); }

    /// @brief lastLeafGrid returns the pointer to the LeafGrid in the cache.
    [[nodiscard]] const LeafGrid* lastLeafGrid() const {
      refreshCache();
      return prev_leaf_ptr_;
    }

    [[nodiscard]] const LeafGrid* getLeafGrid(const CoordT& coord) const {
      return findLeaf(coord.x, coord.y, coord.z);
    }

   protected:
    /// The slow path of the methods above, taken when the coordinates leave the cached
    /// leaf: it goes through the root map. It takes three integers rather than a CoordT,
    /// so that Accessor::findOrCreateLeaf, which must stay out of line, gets them in
    /// registers: see cacheKey() for what a CoordT going through memory costs.
    const LeafGrid* findLeaf(int32_t x, int32_t y, int32_t z) const;

    /// Copies a key into the cache field by field. Copied whole, a CoordT computed in
    /// registers goes through the stack instead: three 32 bits stores, read back with a
    /// 64 bits load for x and y, which cannot be forwarded from those stores and has to
    /// wait for them to retire, that is, for the previous lookup to finish. Random queries
    /// were up to 20% slower that way.
    static void cacheKey(CoordT& cached, const CoordT& key) {
      cached.x = key.x;
      cached.y = key.y;
      cached.z = key.z;
    }

    /// Drop the cache if the grid released the nodes that these pointers refer to.
    void refreshCache() const {
      if (cache_epoch_ == grid_.cache_epoch_.value) {
        return;
      }
      cache_epoch_ = grid_.cache_epoch_.value;
      prev_root_coord_ = {std::numeric_limits<int32_t>::max(), 0, 0};
      prev_inner_coord_ = {std::numeric_limits<int32_t>::max(), 0, 0};
      prev_inner_ptr_ = nullptr;
      prev_leaf_ptr_ = nullptr;
    }

    const VoxelGrid& grid_;
    mutable uint32_t cache_epoch_;
    mutable CoordT prev_root_coord_ = {std::numeric_limits<int32_t>::max(), 0, 0};
    mutable CoordT prev_inner_coord_ = {std::numeric_limits<int32_t>::max(), 0, 0};
    mutable const InnerGrid* prev_inner_ptr_ = nullptr;
    mutable const LeafGrid* prev_leaf_ptr_ = nullptr;
  };

  /** Class to be used to set and get values of a cell of the Grid.
   *  It uses caching to speed up computation.
   *
   *  Create an instance of this object with the method VoxelGrid::createAccessor()
   */
  class Accessor : public ConstAccessor {
   public:
    Accessor(VoxelGrid& grid) : ConstAccessor(grid), mutable_grid_(grid) {}

    /**
     * @brief setValue of a cell. If the cell did not exist, it is created.
     *
     * @param coord   coordinate of the cell
     * @param value   value to set.
     * @return        the previous state of the cell (ON = true).
     */
    bool setValue(const CoordT& coord, const DataT& value);

    /** @brief value getter.
     *
     * @param coord   coordinate of the cell.
     * @return        return the pointer to the value or nullptr if not set.
     */
    [[nodiscard]] DataT* value(const CoordT& coord, bool create_if_missing = false);

    /** @brief setCellOn is similar to setValue, but the value is changed only if the
     * cell has been created, otherwise, the previous value is used.
     *
     * @param coord           coordinate of the cell.
     * @param default_value   default value of the cell. Use only if the cell did not
     * exist before.
     * @return                the previous state of the cell (ON = true).
     */
    bool setCellOn(const CoordT& coord, const DataT& default_value = DataT());

    /** @brief setCellOff will disable a cell without deleting its content.
     *
     * @param coord   coordinate of the cell.
     * @return        the previous state of the cell (ON = true).
     */
    bool setCellOff(const CoordT& coord);

    /**
     * @brief getLeafGrid gets the pointer to the LeafGrid containing the cell.
     * It is the basic class used by setValue() and value().
     *
     * @param coord               Coordinate of the cell.
     * @param create_if_missing   if true, create the Root, Inner and Leaf, if not
     * present.
     */
    [[nodiscard]] LeafGrid* getLeafGrid(const CoordT& coord, bool create_if_missing = false) {
      return findOrCreateLeaf(coord.x, coord.y, coord.z, create_if_missing);
    }

   private:
    /// ConstAccessor::findLeaf, creating the leaf if it is missing and create_if_missing.
    /// Never inlined: creating roots and leaves takes a lot of code, which would make
    /// setValue() and the others too big to be inlined into the loops that call them.
    BONXAI_NOINLINE LeafGrid* findOrCreateLeaf(int32_t x, int32_t y, int32_t z,
                                               bool create_if_missing);

    // The cache lives in ConstAccessor, which is a dependent base class.
    using ConstAccessor::cacheKey;
    using ConstAccessor::prev_inner_coord_;
    using ConstAccessor::prev_inner_ptr_;
    using ConstAccessor::prev_leaf_ptr_;
    using ConstAccessor::prev_root_coord_;
    using ConstAccessor::refreshCache;

    // Since this class is built from a non-const VoxelGrid, the nodes that the cache
    // points to are not const either.
    [[nodiscard]] static LeafGrid* mutableLeaf(const LeafGrid* leaf) {
      return const_cast<LeafGrid*>(leaf);
    }

    VoxelGrid& mutable_grid_;
  };

  Accessor createAccessor() { return Accessor(*this); }

  ConstAccessor createConstAccessor() const { return ConstAccessor(*this); }

  [[nodiscard]] CoordT getRootKey(const CoordT& coord) const;

  [[nodiscard]] CoordT getInnerKey(const CoordT& coord) const;

  [[nodiscard]] uint32_t getInnerIndex(const CoordT& coord) const;

  [[nodiscard]] uint32_t getLeafIndex(const CoordT& coord) const;

 public:
  // Prefer Clear()/releaseUnusedMemory() when deleting nodes: these refresh accessors.
  // Direct erasure through root_map requires recreating existing accessors.
  RootMap root_map;
};

using BinaryVoxelGrid = VoxelGrid<EmptyVoxel>;

//----------------------------------------------------
//----------------- Implementations ------------------
//----------------------------------------------------

template <typename DataT>
inline Grid<DataT>::Grid(Grid&& other)
    : dim_(other.dim_),
      size_(other.size_),
      data_(std::exchange(other.data_, nullptr)),
      mask_(std::move(other.mask_)),
      external_memory_(other.external_memory_) {}

template <typename DataT>
inline Grid<DataT>& Grid<DataT>::operator=(Grid&& other) {
  if (this != &other) {
    if (!external_memory_) {
      delete[] data_;
    }
    dim_ = other.dim_;
    size_ = other.size_;
    data_ = std::exchange(other.data_, nullptr);
    mask_ = std::move(other.mask_);
    external_memory_ = other.external_memory_;
  }
  return *this;
}

template <typename DataT>
inline size_t Grid<DataT>::memUsage() const {
  auto mem = mask_.memUsage() + sizeof(uint8_t) + sizeof(uint32_t) + sizeof(DataT*);
  if (!std::is_same_v<DataT, EmptyVoxel> && !external_memory_) {
    mem += sizeof(DataT) * size_;
  }
  return mem;
}

template <typename DataT, typename Shape>
inline void VoxelGrid<DataT, Shape>::releaseUnusedMemory() {
  // first: if anything below throws, the accessors must not use the leaves already freed
  ++cache_epoch_.value;
  std::vector<CoordT> keys_to_delete;
  for (auto& [key, inner_grid] : root_map) {
    for (auto inner_it = inner_grid.mask().beginOn(); inner_it; ++inner_it) {
      const int32_t inner_index = *inner_it;
      auto& leaf_grid = inner_grid.cell(inner_index);
      if (leaf_grid->mask().isOff()) {
        inner_grid.mask().setOff(inner_index);
        leaf_grid.reset();
      }
    }
    if (inner_grid.mask().isOff()) {
      keys_to_delete.push_back(key);
    }
  }
  for (const auto& key : keys_to_delete) {
    root_map.erase(key);
  }
  leaf_block_allocator_.releaseUnusedMemory();
}

template <typename DataT, typename Shape>
inline VoxelGrid<DataT, Shape>::VoxelGrid(double voxel_size, uint8_t inner_bits, uint8_t leaf_bits)
    : Shape(inner_bits, leaf_bits),
      resolution(voxel_size),
      inv_resolution(1.0 / resolution),
      leaf_block_allocator_(this->LEAF_BITS) {}

template <typename DataT, typename Shape>
inline CoordT VoxelGrid<DataT, Shape>::posToCoord(double x, double y, double z) const {
  return PosToCoord({x, y, z}, inv_resolution);
}

template <typename DataT, typename Shape>
inline Point3D VoxelGrid<DataT, Shape>::coordToPos(const CoordT& coord) const {
  return CoordToPos(coord, resolution);
}

template <typename DataT, typename Shape>
inline CoordT VoxelGrid<DataT, Shape>::getRootKey(const CoordT& coord) const {
  const int32_t MASK = ~((1 << this->LOG2_N) - 1);
  return {coord.x & MASK, coord.y & MASK, coord.z & MASK};
}

template <typename DataT, typename Shape>
inline CoordT VoxelGrid<DataT, Shape>::getInnerKey(const CoordT& coord) const {
  const int32_t MASK = ~((1 << this->LEAF_BITS) - 1);
  return {coord.x & MASK, coord.y & MASK, coord.z & MASK};
}

template <typename DataT, typename Shape>
inline uint32_t VoxelGrid<DataT, Shape>::getInnerIndex(const CoordT& coord) const {
  // clang-format off
  const uint32_t leaf_bits = this->LEAF_BITS;
  const uint32_t inner_bits = this->INNER_BITS;
  const uint32_t inner_mask = this->INNER_MASK;
  return ((coord.x >> leaf_bits) & inner_mask) |
         (((coord.y >> leaf_bits) & inner_mask) << inner_bits) |
         (((coord.z >> leaf_bits) & inner_mask) << (inner_bits * 2));
  // clang-format on
}

template <typename DataT, typename Shape>
inline uint32_t VoxelGrid<DataT, Shape>::getLeafIndex(const CoordT& coord) const {
  // clang-format off
  const uint32_t leaf_bits = this->LEAF_BITS;
  const uint32_t leaf_mask = this->LEAF_MASK;
  return (coord.x & leaf_mask) |
         ((coord.y & leaf_mask) << leaf_bits) |
         ((coord.z & leaf_mask) << (leaf_bits * 2));
  // clang-format on
}

template <typename DataT, typename Shape>
inline bool VoxelGrid<DataT, Shape>::Accessor::setValue(const CoordT& coord, const DataT& value) {
  static_assert(!std::is_same_v<DataT, EmptyVoxel>,
                "You can not access a value when using type EmptyVoxel. Use "
                "setCellOn / setCellOff");

  refreshCache();
  const CoordT inner_key = mutable_grid_.getInnerKey(coord);
  if (inner_key != prev_inner_coord_ || prev_leaf_ptr_ == nullptr) {
    prev_leaf_ptr_ = getLeafGrid(coord, true);
    cacheKey(prev_inner_coord_, inner_key);
  }

  const uint32_t index = mutable_grid_.getLeafIndex(coord);
  LeafGrid* leaf_ptr = mutableLeaf(prev_leaf_ptr_);
  const bool was_on = leaf_ptr->mask().setOn(index);
  leaf_ptr->cell(index) = value;
  return was_on;
}

//----------------------------------
template <typename DataT, typename Shape>
inline DataT* VoxelGrid<DataT, Shape>::Accessor::value(const CoordT& coord,
                                                       bool create_if_missing) {
  static_assert(!std::is_same_v<DataT, EmptyVoxel>,
                "You can not access a value when using type EmptyVoxel. Use "
                "isCellOn / setCellOn / setCellOff");

  refreshCache();
  const CoordT inner_key = mutable_grid_.getInnerKey(coord);

  if (inner_key != prev_inner_coord_ || prev_leaf_ptr_ == nullptr) {
    prev_leaf_ptr_ = getLeafGrid(coord, create_if_missing);
    cacheKey(prev_inner_coord_, inner_key);
  }

  if (prev_leaf_ptr_) {
    LeafGrid* leaf_ptr = mutableLeaf(prev_leaf_ptr_);
    const uint32_t index = mutable_grid_.getLeafIndex(coord);
    if (leaf_ptr->mask().isOn(index)) {
      return &(leaf_ptr->cell(index));
    } else if (create_if_missing) {
      leaf_ptr->mask().setOn(index);
      leaf_ptr->cell(index) = {};
      return &(leaf_ptr->cell(index));
    }
  }
  return nullptr;
}

template <typename DataT, typename Shape>
inline const DataT* VoxelGrid<DataT, Shape>::ConstAccessor::value(const CoordT& coord) const {
  static_assert(!std::is_same_v<DataT, EmptyVoxel>,
                "You can not access a value when using type EmptyVoxel. Use isCellOn");

  refreshCache();
  const CoordT inner_key = grid_.getInnerKey(coord);

  // a cached nullptr must not be trusted: the leaf may have been created since then
  if (inner_key != prev_inner_coord_ || prev_leaf_ptr_ == nullptr) {
    prev_leaf_ptr_ = getLeafGrid(coord);
    cacheKey(prev_inner_coord_, inner_key);
  }

  if (prev_leaf_ptr_) {
    const uint32_t index = grid_.getLeafIndex(coord);
    if (prev_leaf_ptr_->mask().isOn(index)) {
      return &(prev_leaf_ptr_->cell(index));
    }
  }
  return nullptr;
}

template <typename DataT, typename Shape>
inline bool VoxelGrid<DataT, Shape>::ConstAccessor::isCellOn(const CoordT& coord) const {
  refreshCache();
  const CoordT inner_key = grid_.getInnerKey(coord);

  // a cached nullptr must not be trusted: the leaf may have been created since then
  if (inner_key != prev_inner_coord_ || prev_leaf_ptr_ == nullptr) {
    prev_leaf_ptr_ = getLeafGrid(coord);
    cacheKey(prev_inner_coord_, inner_key);
  }

  if (prev_leaf_ptr_) {
    const uint32_t index = grid_.getLeafIndex(coord);
    if (prev_leaf_ptr_->mask().isOn(index)) {
      return true;
    }
  }
  return false;
}

//----------------------------------
template <typename DataT, typename Shape>
inline bool VoxelGrid<DataT, Shape>::Accessor::setCellOn(const CoordT& coord,
                                                         const DataT& default_value) {
  refreshCache();
  const CoordT inner_key = mutable_grid_.getInnerKey(coord);

  if (inner_key != prev_inner_coord_ || prev_leaf_ptr_ == nullptr) {
    prev_leaf_ptr_ = getLeafGrid(coord, true);
    cacheKey(prev_inner_coord_, inner_key);
  }
  LeafGrid* leaf_ptr = mutableLeaf(prev_leaf_ptr_);
  uint32_t index = mutable_grid_.getLeafIndex(coord);
  bool was_on = leaf_ptr->mask().setOn(index);
  if constexpr (!std::is_same_v<DataT, EmptyVoxel>) {
    if (!was_on) {
      leaf_ptr->cell(index) = default_value;
    }
  }
  return was_on;
}

//----------------------------------
template <typename DataT, typename Shape>
inline bool VoxelGrid<DataT, Shape>::Accessor::setCellOff(const CoordT& coord) {
  refreshCache();
  const CoordT inner_key = mutable_grid_.getInnerKey(coord);

  // a leaf missing when it was cached may have been created since, by another accessor
  if (inner_key != prev_inner_coord_ || prev_leaf_ptr_ == nullptr) {
    prev_leaf_ptr_ = getLeafGrid(coord, false);
    cacheKey(prev_inner_coord_, inner_key);
  }
  if (prev_leaf_ptr_) {
    uint32_t index = mutable_grid_.getLeafIndex(coord);
    return mutableLeaf(prev_leaf_ptr_)->mask().setOff(index);
  }
  return false;
}

//----------------------------------
template <typename DataT, typename Shape>
inline typename std::shared_ptr<Grid<DataT>> VoxelGrid<DataT, Shape>::allocateLeafGrid() {
  if constexpr (std::is_trivial_v<DataT> && !std::is_same_v<DataT, EmptyVoxel> &&
                alignof(DataT) <= alignof(std::max_align_t)) {
    auto allocated = leaf_block_allocator_.allocateBlock();
    DataT* memory_block = allocated.first;
    auto deleter = [deleter_impl = std::move(allocated.second)](LeafGrid* ptr) {
      // `delete` runs the destructor: calling it explicitly too destroyed the
      // LeafGrid twice, double-freeing Mask::words_ when leaf_bits >= 4
      deleter_impl();
      delete ptr;
    };
    return std::shared_ptr<LeafGrid>(new LeafGrid(this->LEAF_BITS, memory_block),
                                     std::move(deleter));
  } else {
    return std::make_shared<LeafGrid>(this->LEAF_BITS);
  }
}

// not declared inline, which would contradict BONXAI_NOINLINE: a template needs none
template <typename DataT, typename Shape>
typename VoxelGrid<DataT, Shape>::LeafGrid* VoxelGrid<DataT, Shape>::Accessor::findOrCreateLeaf(
    int32_t x, int32_t y, int32_t z, bool create_if_missing) {
  refreshCache();
  const CoordT coord{x, y, z};
  auto* inner_ptr = const_cast<InnerGrid*>(prev_inner_ptr_);
  const CoordT root_key = mutable_grid_.getRootKey(coord);

  if (root_key != prev_root_coord_ || !inner_ptr) {
    auto& root_map = mutable_grid_.root_map;
    auto it = root_map.find(root_key);
    if (it == root_map.end()) {
      if (!create_if_missing) {
        return nullptr;
      }
      it = root_map.try_emplace(root_key, mutable_grid_.INNER_BITS).first;
    }
    inner_ptr = &(it->second);
    // update the cache
    cacheKey(prev_root_coord_, root_key);
    prev_inner_ptr_ = inner_ptr;
  }

  const uint32_t inner_index = mutable_grid_.getInnerIndex(coord);
  auto& inner_data = inner_ptr->cell(inner_index);

  if (create_if_missing) {
    if (!inner_ptr->mask().setOn(inner_index)) {
      inner_data = mutable_grid_.allocateLeafGrid();
    }
  } else {
    if (!inner_ptr->mask().isOn(inner_index)) {
      return nullptr;
    }
  }
  return inner_data.get();
}

template <typename DataT, typename Shape>
inline const typename VoxelGrid<DataT, Shape>::LeafGrid*
VoxelGrid<DataT, Shape>::ConstAccessor::findLeaf(int32_t x, int32_t y, int32_t z) const {
  refreshCache();
  const CoordT coord{x, y, z};
  const InnerGrid* inner_ptr = prev_inner_ptr_;
  const CoordT root_key = grid_.getRootKey(coord);

  if (root_key != prev_root_coord_ || !inner_ptr) {
    auto it = grid_.root_map.find(root_key);
    if (it == grid_.root_map.end()) {
      return nullptr;
    }
    inner_ptr = &(it->second);
    // update the cache
    cacheKey(prev_root_coord_, root_key);
    prev_inner_ptr_ = inner_ptr;
  }

  const uint32_t inner_index = grid_.getInnerIndex(coord);
  const auto& inner_data = inner_ptr->cell(inner_index);

  if (!inner_ptr->mask().isOn(inner_index)) {
    return nullptr;
  }
  return inner_data.get();
}

template <typename DataT, typename Shape>
inline size_t VoxelGrid<DataT, Shape>::memUsage() const {
  // the index of the root map, and the key of each root: inner_grid.memUsage() below
  // counts the rest of each node
  size_t total_size = root_map.memUsage() + root_map.size() * sizeof(CoordT);

  for (const auto& [key, inner_grid] : root_map) {
    total_size += inner_grid.memUsage();
    for (auto inner_it = inner_grid.mask().beginOn(); inner_it; ++inner_it) {
      const int32_t inner_index = *inner_it;
      auto& leaf_grid = inner_grid.cell(inner_index);
      total_size += leaf_grid->memUsage();
    }
  }
  // if we are using the memory pool, leaf_grid->memUsage() did not return the right
  // value
  total_size += leaf_block_allocator_.memUsage();
  return total_size;
}

template <typename DataT, typename Shape>
inline void VoxelGrid<DataT, Shape>::clear(ClearOption opt) {
  if (opt == CLEAR_MEMORY) {
    ++cache_epoch_.value;
    root_map.clear();
    leaf_block_allocator_.clear();
    return;
  }
  for (auto& [key, inner] : root_map) {
    for (auto it = inner.mask().beginOn(); it; ++it) {
      inner.cell(*it)->mask().setOff();
    }
  }
}

template <typename DataT, typename Shape>
inline size_t VoxelGrid<DataT, Shape>::activeCellsCount() const {
  size_t total_size = 0;

  for (const auto& [key, inner_grid] : root_map) {
    for (auto inner_it = inner_grid.mask().beginOn(); inner_it; ++inner_it) {
      const int32_t inner_index = *inner_it;
      auto& leaf_grid = inner_grid.cell(inner_index);
      total_size += leaf_grid->mask().countOn();
    }
  }
  return total_size;
}

//----------------------------------
template <typename DataT, typename Shape>
template <class VisitorFunction>
inline void VoxelGrid<DataT, Shape>::forEachCell(VisitorFunction func) const {
  const int32_t LEAF_BITS = int32_t(this->LEAF_BITS);
  const int32_t INNER_BITS = int32_t(this->INNER_BITS);
  const int32_t MASK_LEAF = ((1 << LEAF_BITS) - 1);
  const int32_t MASK_INNER = ((1 << INNER_BITS) - 1);

  for (auto& map_it : root_map) {
    const auto& [xA, yA, zA] = (map_it.first);
    const InnerGrid& inner_grid = map_it.second;
    const auto& mask2 = inner_grid.mask();

    for (auto inner_it = mask2.beginOn(); inner_it; ++inner_it) {
      const int32_t inner_index = *inner_it;
      const int32_t INNER_BITS_2 = INNER_BITS * 2;
      // clang-format off
      int32_t xB = xA | ((inner_index & MASK_INNER) << LEAF_BITS);
      int32_t yB = yA | (((inner_index >> INNER_BITS) & MASK_INNER) << LEAF_BITS);
      int32_t zB = zA | (((inner_index >> (INNER_BITS_2)) & MASK_INNER) << LEAF_BITS);
      // clang-format on

      const auto& leaf_grid = inner_grid.cell(inner_index);
      const auto& mask1 = leaf_grid->mask();

      for (auto leaf_it = mask1.beginOn(); leaf_it; ++leaf_it) {
        const int32_t leaf_index = *leaf_it;
        const int32_t LEAF_BITS_2 = LEAF_BITS * 2;
        CoordT pos = {xB | (leaf_index & MASK_LEAF), yB | ((leaf_index >> LEAF_BITS) & MASK_LEAF),
                      zB | ((leaf_index >> (LEAF_BITS_2)) & MASK_LEAF)};
        // apply the visitor
        if constexpr (std::is_same_v<DataT, EmptyVoxel>) {
          EmptyVoxel dummy{};
          func(dummy, pos);
        } else {
          func(leaf_grid->cell(leaf_index), pos);
        }
      }
    }
  }
}

}  // namespace Bonxai

namespace Bonxai {

// Legacy 26-connected integer Bresenham traversal. Includes the origin voxel,
// excludes the endpoint, and permits early termination by the visitor.
template <class Functor>
inline void RayIterator(const CoordT& origin, const CoordT& end, const Functor& func) {
  if (origin == end || !func(origin)) {
    return;
  }
  CoordT coord = origin;
  const int32_t sx = end.x < origin.x ? -1 : 1;
  const int32_t sy = end.y < origin.y ? -1 : 1;
  const int32_t sz = end.z < origin.z ? -1 : 1;
  const int64_t dx = std::abs(int64_t(end.x) - origin.x);
  const int64_t dy = std::abs(int64_t(end.y) - origin.y);
  const int64_t dz = std::abs(int64_t(end.z) - origin.z);
  const int64_t length = std::max({dx, dy, dz});
  // Shift the error by ceil(length / 2): testing its sign is equivalent to
  // the legacy 2 * error >= length, without a multiply or a signed left shift.
  const int64_t half = length / 2 + length % 2;
  int64_t ex = -half, ey = -half, ez = -half;
  for (int64_t i = 0; i < length - 1; ++i) {
    ex += dx;
    ey += dy;
    ez += dz;
    if (ex >= 0) {
      coord.x += sx;
      ex -= length;
    }
    if (ey >= 0) {
      coord.y += sy;
      ey -= length;
    }
    if (ez >= 0) {
      coord.z += sz;
      ez -= length;
    }
    if (!func(coord)) {
      return;
    }
  }
}

inline void ComputeRay(const CoordT& origin, const CoordT& end, std::vector<CoordT>& ray) {
  ray.clear();
  RayIterator(origin, end, [&ray](const CoordT& coord) {
    ray.push_back(coord);
    return true;
  });
}

// Exact traversal from the continuous origin to the endpoint voxel's centre.
// The origin voxel is included and the endpoint voxel is excluded.
template <class PointT, class Functor>
inline void ExactRayIterator(const PointT& origin, const CoordT& coord_from, const CoordT& coord_to,
                             double resolution, const Functor& func) {
  const auto from = ConvertPoint<Point3D>(origin);
  if (coord_from == coord_to) {
    return;
  }
  if (!func(coord_from)) {
    return;
  }

  constexpr double kInf = std::numeric_limits<double>::infinity();

  // scalars, not arrays: a dynamic index would keep t_max/coord in memory
  int32_t cx = coord_from.x;
  int32_t cy = coord_from.y;
  int32_t cz = coord_from.z;
  int32_t sx, sy, sz;
  double tx, ty, tz;
  double dx, dy, dz;

  // parametrized along the unnormalized segment: t = 1 at the endpoint center
  auto setup = [&](double delta, double origin, int32_t coord, int32_t& step, double& t_max,
                   double& t_delta) {
    if (delta != 0.0) {
      const double inv_delta = 1.0 / delta;
      step = (delta > 0.0) ? 1 : -1;
      const double boundary = (double(coord) + (delta > 0.0 ? 1.0 : 0.0)) * resolution;
      t_max = (boundary - origin) * inv_delta;
      t_delta = resolution * std::abs(inv_delta);
    } else {
      step = 0;
      t_max = kInf;
      t_delta = kInf;
    }
  };
  setup((coord_to.x + 0.5) * resolution - from.x, from.x, cx, sx, tx, dx);
  setup((coord_to.y + 0.5) * resolution - from.y, from.y, cy, sy, ty, dy);
  setup((coord_to.z + 0.5) * resolution - from.z, from.z, cz, sz, tz, dz);

  while (true) {
    // advance along the axis with the closest boundary crossing
    if (tx < ty) {
      if (tx < tz) {
        if (tx > 1.0) {
          return;  // no boundary crossing left before the end of the segment
        }
        cx += sx;
        tx += dx;
      } else {
        if (tz > 1.0) {
          return;
        }
        cz += sz;
        tz += dz;
      }
    } else if (ty < tz) {
      if (ty > 1.0) {
        return;
      }
      cy += sy;
      ty += dy;
    } else {
      if (tz > 1.0) {
        return;
      }
      cz += sz;
      tz += dz;
    }
    if (cx == coord_to.x && cy == coord_to.y && cz == coord_to.z) {
      return;  // the endpoint voxel is excluded
    }
    if (!func(CoordT{cx, cy, cz})) {
      return;
    }
  }
}

template <class PointT>
inline void ComputeExactRay(const PointT& origin, const CoordT& coord_from, const CoordT& coord_to,
                            double resolution, std::vector<CoordT>& ray) {
  ray.clear();
  ExactRayIterator(origin, coord_from, coord_to, resolution, [&ray](const CoordT& coord) {
    ray.push_back(coord);
    return true;
  });
}

}  // namespace Bonxai
