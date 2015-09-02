/******************************************************************************
* Copyright (c) 2011, Michael P. Gerlek (mpg@flaxen.com)
* Copyright (c) 2015, Howard Butler (howard@hobu.co)
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/

#include <pdal/plang/Array.hpp>
#include <pdal/plang/Environment.hpp>

#include <algorithm>

#ifdef PDAL_COMPILER_MSVC
#  pragma warning(disable: 4127) // conditional expression is constant
#endif

#include <Python.h>
#undef toupper
#undef tolower
#undef isspace


#define PY_ARRAY_UNIQUE_SYMBOL PDALARRAY_ARRAY_API
#include <numpy/arrayobject.h>

namespace pdal
{
namespace plang
{

Array::Array()
{
    auto initNumpy = []()
    {
#undef NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_RETVAL
        import_array();
    };
    initNumpy();
}


Array::~Array()
{
    cleanup();
}

void Array::cleanup()
{
    for (auto i: m_py_arrays)
    {
        PyObject* p = (PyObject*)(i.second);
        Py_XDECREF(p);
    }

    for (auto& i: m_data_arrays)
    {
        i.second.reset(nullptr);
    }
    m_py_arrays.clear();
    m_data_arrays.clear();
}
std::string Array::buildNumpyDescription() const
{
    std::string output;
    std::stringstream oss;
    Dimension::IdList dims = m_layout->dims();

    for (Dimension::IdList::size_type i=0; i < dims.size(); ++i)
    {
        Dimension::Id::Enum id = (dims[i]);
        Dimension::Type::Enum t = m_layout->dimType(id);
        const int pyDataType = getPythonDataType(t);

        PyArray_Descr *dtype = PyArray_DescrNewFromType(pyDataType);
        Py_INCREF(dtype);
        npy_intp stride = m_layout->dimSize(id);

        std::string name = m_layout->dimName(id);
    }





    return output;
}
void Array::update(PointViewPtr view)
{
    typedef std::unique_ptr<std::vector<uint8_t>> DataPtr;
    cleanup();
    int nd = 1;
    m_layout = view->layout();
    Dimension::IdList dims = m_layout->dims();
    npy_intp mydims = view->size();
    npy_intp* ndims = &mydims;

    for (Dimension::IdList::size_type i=0; i < dims.size(); ++i)
    {
        Dimension::Id::Enum id = (dims[i]);
        Dimension::Type::Enum t = m_layout->dimType(id);

        npy_intp stride = m_layout->dimSize(id);
        npy_intp* strides = &stride;

        std::string name = m_layout->dimName(id);
        std::cout << "name: " << name << " size: " << stride << std::endl;

        DataPtr pdata( new std::vector<uint8_t>(stride * view->size(), 0));
        uint8_t* sp = pdata.get()->data();

        uint8_t* p(sp);
        for (PointId idx = 0; idx < view->size(); ++idx)
        {
            view->getRawField(id, idx, p);
            p += stride;
        }

        int flags = NPY_CARRAY; // NPY_BEHAVED
        const int pyDataType = getPythonDataType(t);
        PyObject* pyArray = PyArray_SimpleNewFromData(nd, ndims, pyDataType, sp);
        PyObject* pyName = PyBytes_FromString(name.c_str());
        PyObject* pyAttrName = PyBytes_FromString("name");
        int did_set = PyObject_SetItem(pyArray, pyAttrName, pyName);
        std::cout << "did_set: " << did_set << std::endl;
//         PyObject* pyArray = PyArray_New(&PyArray_Type, nd, ndims, pyDataType,
//             strides, tp, 0, flags, NULL);

        m_py_arrays.insert(std::pair<std::string, PyObject*>(name,pyArray));
        m_data_arrays.insert(std::make_pair((void*)pyArray, std::move(pdata)));
    }

}

void* Array::getArray(std::string const& name) const
{
    auto found = m_py_arrays.find(name);
    if (found != m_py_arrays.end())
        return found->second;
    else
        return 0;
}

std::vector<void*> Array::getPythonArrays() const
{
    std::vector<void*> output;
    for (auto i: m_py_arrays)
    {
        output.push_back(i.second);
    }

    return output;

}
std::vector<std::string> Array::getArrayNames() const
{
    std::vector<std::string> output;
    for (auto i: m_py_arrays)
    {
        output.push_back(i.first);
    }
    return output;
}

void *Array::extractResult(std::string const& name,
    Dimension::Type::Enum t)
{
//     PyObject* xarr = PyDict_GetItemString(m_varsOut, name.c_str());
//     if (!xarr)
//         throw pdal::pdal_error("plang output variable '" + name + "' not found.");
//     if (!PyArray_Check(xarr))
//         throw pdal::pdal_error("Plang output variable  '" + name +
//             "' is not a numpy array");
//
//     PyArrayObject* arr = (PyArrayObject*)xarr;
//
//     npy_intp one = 0;
//     const int pyDataType = getPythonDataType(t);
//     PyArray_Descr *dtype = PyArray_DESCR(arr);
//
//     if (static_cast<uint32_t>(dtype->elsize) != Dimension::size(t))
//     {
//         std::ostringstream oss;
//         oss << "dtype of array has size " << dtype->elsize
//             << " but PDAL dimension '" << name << "' has byte size of "
//             << Dimension::size(t) << " bytes.";
//         throw pdal::pdal_error(oss.str());
//     }
//
//     using namespace Dimension;
//     BaseType::Enum b = Dimension::base(t);
//     if (dtype->kind == 'i' && b != BaseType::Signed)
//     {
//         std::ostringstream oss;
//         oss << "dtype of array has a signed integer type but the " <<
//             "dimension data type of '" << name <<
//             "' is not pdal::Signed.";
//         throw pdal::pdal_error(oss.str());
//     }
//
//     if (dtype->kind == 'u' && b != BaseType::Unsigned)
//     {
//         std::ostringstream oss;
//         oss << "dtype of array has a unsigned integer type but the " <<
//             "dimension data type of '" << name <<
//             "' is not pdal::Unsigned.";
//         throw pdal::pdal_error(oss.str());
//     }
//
//     if (dtype->kind == 'f' && b != BaseType::Floating)
//     {
//         std::ostringstream oss;
//         oss << "dtype of array has a float type but the " <<
//             "dimension data type of '" << name << "' is not pdal::Floating.";
//         throw pdal::pdal_error(oss.str());
//     }
//     return PyArray_GetPtr(arr, &one);
}


void Array::getOutputNames(std::vector<std::string>& names)
{
//     names.clear();
//
//     PyObject *key, *value;
//     Py_ssize_t pos = 0;
//
//     while (PyDict_Next(m_varsOut, &pos, &key, &value))
//     {
//         const char* p(0);
// #if PY_MAJOR_VERSION >= 3
//         p = PyBytes_AsString(PyUnicode_AsUTF8String(key));
// #else
//         p = PyString_AsString(key);
// #endif
//         if (p)
//             names.push_back(p);
//     }
}


int Array::getPythonDataType(Dimension::Type::Enum t)
{
    using namespace Dimension;

    switch (t)
    {
    case Type::Float:
        return PyArray_FLOAT;
    case Type::Double:
        return PyArray_DOUBLE;
    case Type::Signed8:
        return PyArray_BYTE;
    case Type::Signed16:
        return PyArray_SHORT;
    case Type::Signed32:
        return PyArray_INT;
    case Type::Signed64:
        return PyArray_LONGLONG;
    case Type::Unsigned8:
        return PyArray_UBYTE;
    case Type::Unsigned16:
        return PyArray_USHORT;
    case Type::Unsigned32:
        return PyArray_UINT;
    case Type::Unsigned64:
        return PyArray_ULONGLONG;
    default:
        return -1;
    }
    assert(0);

    return -1;
}


bool Array::hasOutputVariable(const std::string& name) const
{
    return true;
//     return (PyDict_GetItemString(m_varsOut, name.c_str()) != NULL);
}


// bool Array::execute()
// {
//     if (!m_bytecode)
//         throw pdal::pdal_error("No code has been compiled");
//
//     Py_INCREF(m_varsIn);
//     Py_INCREF(m_varsOut);
//     m_scriptArgs = PyTuple_New(2);
//     PyTuple_SetItem(m_scriptArgs, 0, m_varsIn);
//     PyTuple_SetItem(m_scriptArgs, 1, m_varsOut);
//
//     m_scriptResult = PyObject_CallObject(m_function, m_scriptArgs);
//     if (!m_scriptResult)
//         throw pdal::pdal_error(getTraceback());
//
//     if (!PyBool_Check(m_scriptResult))
//         throw pdal::pdal_error("User function return value not a boolean type.");
//
//     return (m_scriptResult == Py_True);
// }
//
} // namespace plang
} // namespace pdal

