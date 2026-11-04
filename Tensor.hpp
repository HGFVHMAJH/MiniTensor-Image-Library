#ifndef MINI_TENSOR_HPP
#define MINI_TENSOR_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// A contiguous, row-major, general-purpose tensor.
template <typename T>
class Tensor
{
protected:
    std::vector<std::size_t> shape_;
    std::vector<T> data_;

    static std::size_t checkedElementCount(const std::vector<std::size_t>& shape)
    {
        // An empty shape represents a scalar (one element).
        std::size_t count = 1;
        for (std::size_t dimension : shape)
        {
            if (dimension == 0)
                throw std::invalid_argument("Tensor dimensions must be positive");
            if (count > std::numeric_limits<std::size_t>::max() / dimension)
                throw std::overflow_error("Tensor element count overflows size_t");
            count *= dimension;
        }
        return count;
    }

    std::size_t offsetOf(const std::vector<std::size_t>& indices) const
    {
        if (indices.size() != shape_.size())
            throw std::invalid_argument("Number of indices must equal tensor rank");

        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < shape_.size(); ++axis)
        {
            if (indices[axis] >= shape_[axis])
                throw std::out_of_range("Tensor index is out of range");
            offset = offset * shape_[axis] + indices[axis];
        }
        return offset;
    }

    void requireSameShape(const Tensor<T>& other, const char* operation) const
    {
        if (shape_ != other.shape_)
            throw std::invalid_argument(std::string(operation) + " requires tensors with identical shapes");
    }

public:
    explicit Tensor(const std::vector<std::size_t>& shape)
        : shape_(shape), data_(checkedElementCount(shape), T{})
    {
    }

    Tensor(const std::vector<std::size_t>& shape, const std::vector<T>& values)
        : shape_(shape), data_(values)
    {
        if (data_.size() != checkedElementCount(shape_))
            throw std::invalid_argument("Data length does not match tensor shape");
    }

    Tensor(std::initializer_list<std::size_t> shape)
        : Tensor(std::vector<std::size_t>(shape))
    {
    }

    virtual ~Tensor() = default;

    std::size_t rank() const noexcept { return shape_.size(); }
    std::size_t size() const noexcept { return data_.size(); }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }

    const std::vector<T>& data() const noexcept { return data_; }
    std::vector<T>& data() noexcept { return data_; }

    void fill(const T& value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }

    void fillTensor(const std::vector<T>& values)
    {
        if (values.size() != data_.size())
            throw std::invalid_argument("Input data length does not match tensor size");
        data_ = values;
    }

    T getElement(const std::vector<std::size_t>& indices) const
    {
        return data_[offsetOf(indices)];
    }

    T getElement(std::initializer_list<std::size_t> indices) const
    {
        return getElement(std::vector<std::size_t>(indices));
    }

    void setElement(const T& value, const std::vector<std::size_t>& indices)
    {
        data_[offsetOf(indices)] = value;
    }

    void setElement(const T& value, std::initializer_list<std::size_t> indices)
    {
        setElement(value, std::vector<std::size_t>(indices));
    }

    // Reshape preserves the row-major element order and does not move the data.
    void reshape(const std::vector<std::size_t>& newShape)
    {
        if (checkedElementCount(newShape) != data_.size())
            throw std::invalid_argument("Reshape must preserve the number of elements");
        shape_ = newShape;
    }

    Tensor<T> operator+(const Tensor<T>& other) const
    {
        requireSameShape(other, "Addition");
        Tensor<T> result(shape_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] + other.data_[i];
        return result;
    }

    Tensor<T> operator-(const Tensor<T>& other) const
    {
        requireSameShape(other, "Subtraction");
        Tensor<T> result(shape_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] - other.data_[i];
        return result;
    }

    // Element-wise (Hadamard) multiplication, not matrix multiplication.
    Tensor<T> operator*(const Tensor<T>& other) const
    {
        requireSameShape(other, "Element-wise multiplication");
        Tensor<T> result(shape_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] * other.data_[i];
        return result;
    }
};

#endif
