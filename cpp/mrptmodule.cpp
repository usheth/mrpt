/*
 * This file wraps the C++11 Mrpt code to an extension module compatible with
 * Python 2.7.
 */

#include "Python.h"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "Mrpt.h"
#include "numpy/arrayobject.h"

#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;

typedef struct {
    PyObject_HEAD
    Mrpt *ptr;
    float *data;
    bool mmap;
    int n;
    int dim;
} mrptIndex;

static PyObject *Mrpt_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    mrptIndex *self;
    self = reinterpret_cast<mrptIndex *>(type->tp_alloc(type, 0));
    if (self != NULL) {
        self->ptr = NULL;
        self->data = NULL;
    }
    return reinterpret_cast<PyObject *>(self);
}

float *read_memory(char *file, int n, int dim) {
    float *data = new float[n * dim];

    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL)
        return NULL;

    int read = fread(data, sizeof(float), n * dim, fd);
    if (read != n * dim)
        return NULL;

    fclose(fd);
    return data;
}

#ifndef _WIN32
float *read_mmap(char *file, int n, int dim) {
    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL)
        return NULL;

    float *data;

    if ((data = reinterpret_cast<float *> (
#ifdef MAP_POPULATE
            mmap(0, n * dim * sizeof(float), PROT_READ,
            MAP_SHARED | MAP_POPULATE, fileno(fd), 0))) == MAP_FAILED) {
#else
            mmap(0, n * dim * sizeof(float), PROT_READ,
            MAP_SHARED, fileno(fd), 0))) == MAP_FAILED) {
#endif
            return NULL;
    }

    fclose(fd);
    return data;
}
#endif

static int Mrpt_init(mrptIndex *self, PyObject *args) {
    PyObject *py_data;
    int depth, n_trees, n, dim, mmap;
    float density;

    if (!PyArg_ParseTuple(args, "Oiiiifi", &py_data, &n, &dim, &depth, &n_trees, &density, &mmap))
        return -1;

    float *data;
#if PY_MAJOR_VERSION >= 3
    if (PyUnicode_Check(py_data)) {
        char *file = PyBytes_AsString(py_data);
#else
    if (PyString_Check(py_data)) {
        char *file = PyString_AsString(py_data);
#endif

        struct stat sb;
        if (stat(file, &sb) != 0) {
            PyErr_SetString(PyExc_IOError, strerror(errno));
            return -1;
        }

        if (sb.st_size != static_cast<unsigned>(sizeof(float) * dim * n)) {
            PyErr_SetString(PyExc_ValueError, "Size of the input is not N x dim");
            return -1;
        }

#ifndef _WIN32
        data = mmap ? read_mmap(file, n, dim) : read_memory(file, n, dim);
#else
        data = read_memory(file, n, dim);
#endif

        if (data == NULL) {
            PyErr_SetString(PyExc_IOError, "Unable to read data from file or allocate memory for it");
            return -1;
        }

        self->mmap = mmap;
        self->data = data;
    } else {
        data = reinterpret_cast<float *>(PyArray_DATA(py_data));
    }

    self->n = n;
    self->dim = dim;

    const Eigen::Map<const MatrixXf> *X = new Eigen::Map<const MatrixXf>(data, dim, n);
    self->ptr = new Mrpt(X, n_trees, depth, density);

    return 0;
}

static PyObject *build(mrptIndex *self) {
    self->ptr->grow();
    Py_RETURN_NONE;
}

static void mrpt_dealloc(mrptIndex *self) {
    if (self->data) {
#ifndef _WIN32
        if (self->mmap)
            munmap(self->data, self->n * self->dim * sizeof(float));
        else
#endif
            delete[] self->data;
    }
    if (self->ptr)
        delete self->ptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *get_leaf_info(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int len,dimensions;
    if (!PyArg_ParseTuple(args, "Oii", &v, &len, &dimensions))
        return NULL;
    int *leaf_indices = reinterpret_cast<int *>(PyArray_DATA(v));
    PyObject *leaf_dict = PyDict_New();
    npy_intp dims[1] = {dimensions};
    float *vals;
    PyObject *coordinates;
    PyObject *key;
    for(int i=0;i<len;i++) {
        coordinates = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
        vals = reinterpret_cast<float *>(PyArray_DATA(coordinates));
        self->ptr->get_leaf_info(leaf_indices[i],vals);
        key = PyLong_FromLong((long)leaf_indices[i]);
        PyDict_SetItem(leaf_dict,key,coordinates);
        Py_DECREF(coordinates);
        Py_DECREF(key);
    }
    return leaf_dict;

}

template<typename T>
static PyArrayObject* vector_to_nparray(const std::vector<T>& vec, int type_num = PyArray_FLOAT){

   // rows not empty
   if( !vec.empty() ){

       size_t nRows = vec.size();
       npy_intp dims[1] = {nRows};

       PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, type_num);
       T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

       std::copy(vec.begin(),vec.end(),vec_array_pointer);
       return vec_array;

   // no data at all
   } else {
      npy_intp dims[1] = {0};
      return (PyArrayObject*) PyArray_ZEROS(1, dims, PyArray_FLOAT, 0);
   }

}

static PyArrayObject *get_leaves(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int n,dim;

    if (!PyArg_ParseTuple(args, "O", &v))
        return NULL;

    float *indata = reinterpret_cast<float *>(PyArray_DATA(v));

    std::vector<int> leaf_indices;

    dim = PyArray_DIM(v, 0);
    self->ptr->get_leaf_indices(Eigen::Map<VectorXf>(indata, dim), &leaf_indices);
    PyArrayObject *leaves = vector_to_nparray(leaf_indices,PyArray_INT);
    return leaves;
}

static PyArrayObject *filter_leaves_by_votes(mrptIndex *self, PyObject *args) {
    PyObject *l;

    int num_leaves,votes_required, dim;

    if (!PyArg_ParseTuple(args, "Oii", &l, &num_leaves, &votes_required))
        return NULL;

    VectorXi votes = VectorXi::Zero(num_leaves);
    std::vector<int> voted_leaves;
    int *leaves = reinterpret_cast<int *>(PyArray_DATA(l));
    for(int i=0;i<num_leaves;i++,leaves++) {
        if(++votes(*leaves) == votes_required){
            voted_leaves.push_back(*leaves);
        }
    }

    PyArrayObject *final_leaves = vector_to_nparray(voted_leaves,PyArray_INT);
    return final_leaves;

}

static PyObject *ann_from_leaves(mrptIndex *self, PyObject *args) {
    PyObject *v;
    PyObject *l;
    int k, elect, dim, num_leaves, return_distances;

    if (!PyArg_ParseTuple(args, "OOiiii", &v, &l, &num_leaves, &k, &elect, &return_distances))
        return NULL;

    float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
    const int *leaves = reinterpret_cast<int *>(PyArray_DATA(l));
    PyObject *nearest;
    dim = PyArray_DIM(v, 0);
    npy_intp dims[1] = {k};
    nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));
    if (return_distances) {
        PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
        float *out_distances = reinterpret_cast<float *>(PyArray_DATA(distances));
        self->ptr->query_from_leaves(Eigen::Map<VectorXf>(indata, dim), leaves, num_leaves, k, elect, outdata, out_distances);

        PyObject *out_tuple = PyTuple_New(2);
        PyTuple_SetItem(out_tuple, 0, nearest);
        PyTuple_SetItem(out_tuple, 1, distances);
        return out_tuple;
    } else {
        self->ptr->query_from_leaves(Eigen::Map<VectorXf>(indata, dim), leaves, num_leaves, k, elect, outdata);
        return nearest;
    }

}

static PyObject *ann(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int k, elect, dim, n, return_distances;

    if (!PyArg_ParseTuple(args, "Oiii", &v, &k, &elect, &return_distances))
        return NULL;

    float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
    PyObject *nearest;

    if (PyArray_NDIM(v) == 1) {

        dim = PyArray_DIM(v, 0);

        npy_intp dims[1] = {k};
        nearest = PyArray_SimpleNew(1, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            float *out_distances = reinterpret_cast<float *>(PyArray_DATA(distances));
            self->ptr->query(Eigen::Map<VectorXf>(indata, dim), k, elect, outdata, out_distances);

            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            self->ptr->query(Eigen::Map<VectorXf>(indata, dim), k, elect, outdata);
            return nearest;
        }
    } else {
        n = PyArray_DIM(v, 0);
        dim = PyArray_DIM(v, 1);

        npy_intp dims[2] = {n, k};
        nearest = PyArray_SimpleNew(2, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            npy_intp dims[2] = {n, k};
            PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            float *distances_out = reinterpret_cast<float *>(PyArray_DATA(distances));

            for (int i = 0; i < n; ++i) {
                self->ptr->query(Eigen::Map<VectorXf>(indata + i * dim, dim),
                                 k, elect, outdata + i * k, distances_out + i * k);
            }
            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            for (int i = 0; i < n; ++i) {
                self->ptr->query(Eigen::Map<VectorXf>(indata + i * dim, dim),
                                 k, elect, outdata + i * k);
            }
            return nearest;
        }
    }
}

static PyObject *get_nearest_leaves(mrptIndex *self, PyObject *args) {
    PyObject *v,*leaves;
    int num_leaves,k,dim;
    if (!PyArg_ParseTuple(args, "OOii", &v, &leaves, &num_leaves, &k))
        return NULL;
    float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
    int *leaf_index = reinterpret_cast<int *>(PyArray_DATA(leaves));
    dim = PyArray_DIM(v, 0);
    npy_intp dims[1] = {k};
    PyObject *nearest = PyArray_SimpleNew(1, dims, NPY_INT);
    int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));
    VectorXi idx(num_leaves);
    for(int i=0;i<num_leaves;i++) {
        idx(i) = leaf_index[i];
    }
    self->ptr->exact_knn(Eigen::Map<VectorXf>(indata, dim), k, idx, num_leaves, outdata);
    return nearest;
}

static PyObject *exact_search(mrptIndex *self, PyObject *args) {
    PyObject *v;
    int k, n, dim, return_distances;

    if (!PyArg_ParseTuple(args, "Oii", &v, &k, &return_distances))
        return NULL;

    float *indata = reinterpret_cast<float *>(PyArray_DATA(v));
    PyObject *nearest;

    VectorXi idx(self->n);
    std::iota(idx.data(), idx.data() + self->n, 0);

    if (PyArray_NDIM(v) == 1) {
        dim = PyArray_DIM(v, 0);

        npy_intp dims[1] = {k};
        nearest = PyArray_SimpleNew(1, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            PyObject *distances = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            float *out_distances = reinterpret_cast<float *>(PyArray_DATA(distances));
            self->ptr->exact_knn(Eigen::Map<VectorXf>(indata, dim), k, idx, self->n, outdata, out_distances);

            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            self->ptr->exact_knn(Eigen::Map<VectorXf>(indata, dim), k, idx, self->n, outdata);
            return nearest;
        }
    } else {
        n = PyArray_DIM(v, 0);
        dim = PyArray_DIM(v, 1);

        npy_intp dims[2] = {n, k};
        nearest = PyArray_SimpleNew(2, dims, NPY_INT);
        int *outdata = reinterpret_cast<int *>(PyArray_DATA(nearest));

        if (return_distances) {
            npy_intp dims[2] = {n, k};
            PyObject *distances = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            float *distances_out = reinterpret_cast<float *>(PyArray_DATA(distances));

            for (int i = 0; i < n; ++i) {
                self->ptr->exact_knn(Eigen::Map<VectorXf>(indata + i * dim, dim),
                                     k, idx, self->n, outdata + i * k, distances_out + i * k);
            }
            PyObject *out_tuple = PyTuple_New(2);
            PyTuple_SetItem(out_tuple, 0, nearest);
            PyTuple_SetItem(out_tuple, 1, distances);
            return out_tuple;
        } else {
            for (int i = 0; i < n; ++i) {
                self->ptr->exact_knn(Eigen::Map<VectorXf>(indata + i * dim, dim),
                                     k, idx, self->n, outdata + i * k);
            }
            return nearest;
        }
    }
}

static PyObject *save(mrptIndex *self, PyObject *args) {
    char *fn;

    if (!PyArg_ParseTuple(args, "s", &fn))
        return NULL;

    if (!self->ptr->save(fn)) {
        PyErr_SetString(PyExc_IOError, "Unable to save index to file");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *load(mrptIndex *self, PyObject *args) {
    char *fn;

    if (!PyArg_ParseTuple(args, "s", &fn))
        return NULL;

    if (!self->ptr->load(fn)) {
        PyErr_SetString(PyExc_IOError, "Unable to load index from file");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef MrptMethods[] = {
    {"filter_leaves_by_votes", (PyCFunction) filter_leaves_by_votes, METH_VARARGS,
            "Filters array of leaves by votes required"},
    {"ann", (PyCFunction) ann, METH_VARARGS,
            "Return approximate nearest neighbors"},
    {"ann_from_leaves", (PyCFunction) ann_from_leaves, METH_VARARGS,
            "Return approximate nearest neighbors given only leaves"},
    {"get_leaf_info", (PyCFunction) get_leaf_info, METH_VARARGS,
            "Returns the coordinates of the leaves by index"},
    {"exact_search", (PyCFunction) exact_search, METH_VARARGS,
            "Return exact nearest neighbors"},
    {"build", (PyCFunction) build, METH_VARARGS,
            "Build the index"},
    {"save", (PyCFunction) save, METH_VARARGS,
            "Save the index to a file"},
    {"load", (PyCFunction) load, METH_VARARGS,
            "Load the index from a file"},
    {"get_leaves", (PyCFunction) get_leaves, METH_VARARGS,
            "Returns the leaves for a query point"},
    {"get_nearest_leaves", (PyCFunction) get_nearest_leaves, METH_VARARGS,
            "Returns the leaves for a query point"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject MrptIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "mrpt.MrptIndex", /*tp_name*/
    sizeof(mrptIndex), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) mrpt_dealloc, /*tp_dealloc*/
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Mrpt index object", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    MrptMethods, /* tp_methods */
    0, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) Mrpt_init, /* tp_init */
    0, /* tp_alloc */
    Mrpt_new, /* tp_new */
};

static PyMethodDef module_methods[] = {
  {NULL}	/* Sentinel */
};
  
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mrptlib",          /* m_name */
    "",                  /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
  
PyMODINIT_FUNC PyInit_mrptlib(void) {
    PyObject *m;
    if (PyType_Ready(&MrptIndexType) < 0)
        return NULL;
    
    m = PyModule_Create(&moduledef);
    

    if (m == NULL)
        return NULL;

    import_array();

    Py_INCREF(&MrptIndexType);
    PyModule_AddObject(m, "MrptIndex", reinterpret_cast<PyObject *>(&MrptIndexType));

    return m;
}
#else
PyMODINIT_FUNC initmrptlib(void) {
    PyObject *m;
    if (PyType_Ready(&MrptIndexType) < 0)
        return;

    m = Py_InitModule("mrptlib", module_methods);


    if (m == NULL)
        return;

    import_array();

    Py_INCREF(&MrptIndexType);
    PyModule_AddObject(m, "MrptIndex", reinterpret_cast<PyObject *>(&MrptIndexType));
}
#endif

