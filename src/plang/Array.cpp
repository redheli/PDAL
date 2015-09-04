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
    : m_py_array(0)
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
//     for (auto i: m_py_arrays)
//     {
//         PyObject* p = (PyObject*)(i.second);
//         Py_XDECREF(p);
//     }
//
//     for (auto& i: m_data_arrays)
//     {
//         i.second.reset(nullptr);
//     }
//     m_py_arrays.clear();
//     m_data_arrays.clear();
}
void* Array::buildNumpyDescription(PointViewPtr view) const
{
    std::stringstream oss;
    Dimension::IdList dims = view->dims();

    PyObject* dict = PyDict_New();
    PyObject* sizes = PyList_New(dims.size());
    PyObject* formats = PyList_New(dims.size());
    PyObject* titles = PyList_New(dims.size());

    for (Dimension::IdList::size_type i=0; i < dims.size(); ++i)
    {
        Dimension::Id::Enum id = (dims[i]);
        Dimension::Type::Enum t = view->dimType(id);
        npy_intp stride = view->dimSize(id);

        std::string name = view->dimName(id);

        std::string kind("i");
        Dimension::BaseType::Enum b = Dimension::base(t);
        if (b == Dimension::BaseType::Unsigned)
            kind = "u";
        else if (b == Dimension::BaseType::Floating)
            kind = "f";

        oss << kind << stride;
        PyObject* pySize = PyLong_FromLong(stride);
        PyObject* pyTitle = PyUnicode_FromString(name.c_str());
        PyObject* pyFormat = PyUnicode_FromString(oss.str().c_str());

        PyList_SetItem(sizes, i, pySize);
        PyList_SetItem(titles, i, pyTitle);
        PyList_SetItem(formats, i, pyFormat);

        oss.str("");
    }

    PyDict_SetItemString(dict, "names", titles);
    PyDict_SetItemString(dict, "formats", formats);


    PyObject* obj = PyUnicode_AsASCIIString(PyObject_Str(dict));
    const char* s = PyBytes_AsString(obj);
    std::string output(s);
    std::cout << "array: " << output << std::endl;
    return (void*) dict;
//     return output;
}
void Array::update(PointViewPtr view)
{
    typedef std::unique_ptr<std::vector<uint8_t>> DataPtr;
    cleanup();
    int nd = 1;
    Dimension::IdList dims = view->dims();
    npy_intp mydims = view->size();
    npy_intp* ndims = &mydims;
    std::vector<npy_intp> strides(dims.size());


//     DataPtr pdata( new std::vector<uint8_t>(view->pointSize()* view->size(), 0));
    m_data_array = new std::vector<uint8_t>(view->pointSize()* view->size(), 0);

//     uint8_t* sp = pdata.get()->data();
    uint8_t* sp = m_data_array->data();
    for (Dimension::IdList::size_type i=0; i < dims.size(); ++i)
    {
        Dimension::Id::Enum id = (dims[i]);
        npy_intp stride = view->dimSize(id);
        strides [i] = stride;
    }

    std::cout << "strides: ";
    for (auto s: strides)
    {
        std::cout << " " << s;
    }
    std::cout << std::endl;

    PyArray_Descr *dtype(0);
    PyObject * dtype_list = (PyObject*)buildNumpyDescription(view);
    if (!dtype_list) throw pdal_error("we're nothing!");
//     int did_convert = PyArray_DescrAlignConverter(dtype_list, &dtype);
    int did_convert = PyArray_DescrConverter(dtype_list, &dtype);
    if (did_convert == NPY_FAIL) throw pdal_error("did not convert!");
//     Py_XDECREF(dtype_list);

    int flags = NPY_CARRAY;
//     int flags = NPY_C_CONTIGUOUS;
    PyObject * pyArray = PyArray_NewFromDescr(&PyArray_Type, dtype, nd, ndims, 0, sp, flags, NULL);

// copy the data
//
    uint8_t* p(sp);

    DimTypeList types = view->dimTypes();
    for (PointId idx = 0; idx < view->size(); idx++)
    {
        p = sp + (view->pointSize() * idx);
        view->getPackedPoint(types, idx, (char*)p);
        p += 12;
//         p = sp + (view->pointSize() * idx);
//         for (auto d: dims)
//         {
//             view->getRawField(d, idx, p);
//             p += view->dimSize(d);
//         }

    }

    m_py_array = pyArray;
//     m_data_array = std::move(pdata);

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

