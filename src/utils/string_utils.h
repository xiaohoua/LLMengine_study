template<typename T>
inline std::string vec2str(std::vector<T> vec)
{
    std::stringstream ss;
    ss << "(";
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            ss << vec[i] << ", ";
        }
        ss << vec.back();
    }
    ss << ")";
    return ss.str();
}

template<typename T>
inline std::string arr2str(T* arr, size_t size)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size - 1; ++i) {
        ss << arr[i] << ", ";
    }
    if (size > 0) {
        ss << arr[size - 1];
    }
    ss << ")";
    return ss.str();
}