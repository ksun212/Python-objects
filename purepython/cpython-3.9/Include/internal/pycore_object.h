#ifndef Py_INTERNAL_OBJECT_H
#define Py_INTERNAL_OBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_gc.h"         // _PyObject_GC_IS_TRACKED()
#include "pycore_interp.h"     // PyInterpreterState.gc
#include "pycore_pystate.h"    // _PyThreadState_GET()
#include "pycore_pyerrors.h"
#include "pycore_hashtable.h"
#define MAX_TYPE_REPR 8000
extern char *TARGET_REPO;
extern int write_mutex;
PyAPI_FUNC(int) _PyType_CheckConsistency(PyTypeObject *type);
PyAPI_FUNC(int) _PyDict_CheckConsistency(PyObject *mp, int check_content);

/* Only private in Python 3.10 and 3.9.8+; public in 3.11 */
extern PyObject *_PyType_GetQualName(PyTypeObject *type);

/* Tell the GC to track this object.
 *
 * NB: While the object is tracked by the collector, it must be safe to call the
 * ob_traverse method.
 *
 * Internal note: interp->gc.generation0->_gc_prev doesn't have any bit flags
 * because it's not object header.  So we don't use _PyGCHead_PREV() and
 * _PyGCHead_SET_PREV() for it to avoid unnecessary bitwise operations.
 *
 * The PyObject_GC_Track() function is the public version of this macro.
 */
static inline void _PyObject_GC_TRACK_impl(const char *filename, int lineno,
                                           PyObject *op)
{
    _PyObject_ASSERT_FROM(op, !_PyObject_GC_IS_TRACKED(op),
                          "object already tracked by the garbage collector",
                          filename, lineno, "_PyObject_GC_TRACK");

    PyGC_Head *gc = _Py_AS_GC(op);
    _PyObject_ASSERT_FROM(op,
                          (gc->_gc_prev & _PyGC_PREV_MASK_COLLECTING) == 0,
                          "object is in generation which is garbage collected",
                          filename, lineno, "_PyObject_GC_TRACK");

    PyThreadState *tstate = _PyThreadState_GET();
    PyGC_Head *generation0 = tstate->interp->gc.generation0;
    PyGC_Head *last = (PyGC_Head*)(generation0->_gc_prev);
    _PyGCHead_SET_NEXT(last, gc);
    _PyGCHead_SET_PREV(gc, last);
    _PyGCHead_SET_NEXT(gc, generation0);
    generation0->_gc_prev = (uintptr_t)gc;
}

#define _PyObject_GC_TRACK(op) \
    _PyObject_GC_TRACK_impl(__FILE__, __LINE__, _PyObject_CAST(op))

/* Tell the GC to stop tracking this object.
 *
 * Internal note: This may be called while GC. So _PyGC_PREV_MASK_COLLECTING
 * must be cleared. But _PyGC_PREV_MASK_FINALIZED bit is kept.
 *
 * The object must be tracked by the GC.
 *
 * The PyObject_GC_UnTrack() function is the public version of this macro.
 */
static inline void _PyObject_GC_UNTRACK_impl(const char *filename, int lineno,
                                             PyObject *op)
{
    _PyObject_ASSERT_FROM(op, _PyObject_GC_IS_TRACKED(op),
                          "object not tracked by the garbage collector",
                          filename, lineno, "_PyObject_GC_UNTRACK");

    PyGC_Head *gc = _Py_AS_GC(op);
    PyGC_Head *prev = _PyGCHead_PREV(gc);
    PyGC_Head *next = _PyGCHead_NEXT(gc);
    _PyGCHead_SET_NEXT(prev, next);
    _PyGCHead_SET_PREV(next, prev);
    gc->_gc_next = 0;
    gc->_gc_prev &= _PyGC_PREV_MASK_FINALIZED;
}

#define _PyObject_GC_UNTRACK(op) \
    _PyObject_GC_UNTRACK_impl(__FILE__, __LINE__, _PyObject_CAST(op))

#ifdef Py_REF_DEBUG
extern void _PyDebug_PrintTotalRefs(void);
#endif

#ifdef Py_TRACE_REFS
extern void _Py_AddToAllObjects(PyObject *op, int force);
extern void _Py_PrintReferences(FILE *);
extern void _Py_PrintReferenceAddresses(FILE *);
#endif

static inline PyObject **
_PyObject_GET_WEAKREFS_LISTPTR(PyObject *op)
{
    Py_ssize_t offset = Py_TYPE(op)->tp_weaklistoffset;
    return (PyObject **)((char *)op + offset);
}

// Fast inlined version of PyType_HasFeature()
static inline int
_PyType_HasFeature(PyTypeObject *type, unsigned long feature) {
    return ((type->tp_flags & feature) != 0);
}

// Fast inlined version of PyObject_IS_GC()
static inline int
_PyObject_IS_GC(PyObject *obj)
{
    return (PyType_IS_GC(Py_TYPE(obj))
            && (Py_TYPE(obj)->tp_is_gc == NULL
                || Py_TYPE(obj)->tp_is_gc(obj)));
}

#define FQN(tp, s) \
    do{ \
        PyObject* modulename;\
        _Py_IDENTIFIER(__module__); \
        modulename = _PyObject_GetAttrIdNoError(tp, &PyId___module__); \
        if (modulename == NULL || !PyUnicode_Check(modulename)) \
        {\
            s = "unknown_module"; \
        } \
        else{ \
            s = PyUnicode_AsUTF8(modulename);  \
            if(s == NULL){ \
                s = "unknown_module"; \
            } \
        } \
        Py_XDECREF(modulename); \
    } while (0)

// #define FQN(tp, s) \
//     do{ \
//         s = "<unknown-module???";\
//     } while (0)
static Py_uhash_t
_Py_hashtable_hash_sizet(const size_t key)
{
    return (Py_uhash_t) key;
}
static int
hashtable_compare_sizet(const size_t key1, const size_t key2)
{
    if (key1 == key2){
        return 1;
    }
    else{
        return 0;
    }
}

// static int init_interesting();
// static void add_interesting_dict(void * dic);
// static void remove_interesting_dict(size_t dic);
// static int query_interesting_dict(size_t dic);
// static int init_interesting(){
//     _Py_hashtable_allocator_t hashtable_alloc = {malloc, free};
//     interesting_dict = _Py_hashtable_new_full(_Py_hashtable_hash_sizet,
//                                           hashtable_compare_sizet,
//                                           NULL, NULL, &hashtable_alloc);
// }
// static void add_interesting_dict(void * dic){
//     if(interesting_dict -> nentries > 1000){
//         return;
//     }
//     _Py_hashtable_entry_t * entry = _Py_hashtable_get_entry(interesting_dict, dic);
//     if (entry != NULL) {
//         ;
//     }
//     else{
//         if (_Py_hashtable_set(interesting_dict, dic, (void *)(uintptr_t) 1) < 0) {
//             ;
//         }
//     }
// }
// static void remove_interesting_dict(size_t dic){
//     _Py_hashtable_entry_t * entry = _Py_hashtable_get_entry(interesting_dict, dic);
//     if (entry != NULL) {
//         entry -> value = NULL;
//     }
//     else{
//         if (_Py_hashtable_set(interesting_dict, (void*) dic, NULL) < 0) {
//             ;
//         }
//     }
// }
// static int query_interesting_dict(size_t dic){
//     _Py_hashtable_entry_t * entry = _Py_hashtable_get_entry(interesting_dict, dic);
//     if (entry != NULL) {
//         int live = (int)(uintptr_t)entry->value;
//         if(live == 1){
//             return 1;
//         }
//         else{
//             return 0;
//         }
//     }
//     else{
//         return 0;
//     }
    
// }

#define FRAME_ID() \
    do{ \
        co_filename = PyUnicode_AsUTF8(f->f_code->co_filename); \
        if (!co_filename) \
            co_filename = "?"; \
        co_name = PyUnicode_AsUTF8(f->f_code->co_name); \
        if (!co_name)\
            co_name = "?"; \
    } while(0)

#define FRAME_ID_SP() \
    do{ \
        co_filename = PyUnicode_AsUTF8(frame->f_code->co_filename); \
        if (!co_filename) \
            co_filename = "?"; \
        co_name = PyUnicode_AsUTF8(frame->f_code->co_name); \
        if (!co_name)\
            co_name = "?"; \
    } while(0)

#define OBJ_TRACE(dic) \
    do{ \
        fprintf(fp, "<opcode> <obj>:<%p>: %s: %s-%p: %d\n", \
            (void *) dic, co_filename, co_name, (void*) f, PyFrame_GetLineNumber(f)); \ 
    } while (0)
    
static int testing(){
    return strcmp(TARGET_REPO, "Nothing") != 0;
}

static void attr_fi(char * fi){
    strcpy(fi, "/home/user/purepython/cpython-3.9/pydyna/store_attr_flow_");
    strcat(fi, TARGET_REPO);
    strcat(fi, ".txt");
}
static void mro_fi(char * fi){
    strcpy(fi, "/home/user/purepython/cpython-3.9/pydyna/store_attr_mro_");
    strcat(fi, TARGET_REPO);
    strcat(fi, ".txt");
}
static void attr_load(char * fi){
    strcpy(fi, "/home/user/purepython/cpython-3.9/pydyna/store_attr_load_");
    strcat(fi, TARGET_REPO);
    strcat(fi, ".txt");
}

static void attr_err(char * fi){
    strcpy(fi, "/home/user/purepython/cpython-3.9/pydyna/store_attr_flow_err_");
    strcat(fi, TARGET_REPO);
    strcat(fi, ".txt");
}

static void attrc_fi(char * fi){
    strcpy(fi, "/home/user/purepython/cpython-3.9/pydyna/store_attr_flowc_");
    strcat(fi, TARGET_REPO);
    strcat(fi, ".txt");
}
static int BaseType(PyObject * obj){
    // return 1;
    char * t;
    FQN(obj->ob_type, t);
    int targeting = (strstr(t, "builtin") != NULL);
    return targeting;
}
static int Type_Of_Interest(PyObject * obj){
    char * t;
    FQN(obj->ob_type, t);
    char repr[1000];
    strcpy(repr, t);
    strcat(repr, ".");
    strcat(repr, obj->ob_type -> tp_name);
    int targeting = (strstr(repr, TARGET_REPO) != NULL);
    // int targeting = (strstr(repr, "cookiecutter.environment.StrictEnvironment") != NULL);
    
    return targeting;
}


static void _Type_Representation(PyObject * obj, char * repr, int class_repr, PyObject * visited_obj){
    
    if(strstr(obj->ob_type -> tp_name, "SimpleNamespace") != NULL){
        strcpy(repr, obj->ob_type -> tp_name);    
    }
    else{
        char * t;
        FQN(obj->ob_type, t);
        strcpy(repr, t);
        strcat(repr, ".");
        strcat(repr, obj->ob_type -> tp_name);

        // mro         
        if(query_interesting_class((void*) obj->ob_type)){
            ;
        }
        else{
            add_interesting_class((void*) obj->ob_type);
            char afi[1005];
            mro_fi(afi);
            FILE * fp = fopen(afi, "a+");
            PyTypeObject *type = obj->ob_type;
            PyObject *mro, *base, *dict;
            mro = type->tp_mro;

            if (mro != NULL) {
                char base_class_name[MAX_TYPE_REPR] = {0};
                char * t;
                FQN(obj->ob_type, t);
                strcat(base_class_name, t);
                strcat(base_class_name, ".");
                strcat(base_class_name, type -> tp_name);
                assert(PyTuple_Check(mro));
                int n = PyTuple_GET_SIZE(mro);
                int i;
                char class_repr[10*MAX_TYPE_REPR] = {0};
                for (i = 0; i < n; i++) {
                    base = PyTuple_GET_ITEM(mro, i);
                    char class_name[MAX_TYPE_REPR] = {0};
                    char * t;
                    FQN(base, t);
                    strcat(class_name, t);
                    strcat(class_name, ".");
                    strcat(class_name, ((PyTypeObject *)base) -> tp_name);
                    assert(PyType_Check(base));
                    dict = ((PyTypeObject *)base)->tp_dict;
                    assert(dict && PyDict_Check(dict));
                    char dict_repr[MAX_TYPE_REPR] = {0};
                    if(strlen(repr) + 10 < MAX_TYPE_REPR){
                            strcat(dict_repr, "{");
                    }
                    if(dict){
                        PyObject *key;
                        PyObject *keys = NULL, *keys_iter = NULL;
                        Py_INCREF(dict);
                        keys = PyDict_Keys(dict);
                        if (keys != NULL){
                            keys_iter = PyObject_GetIter(keys);
                            if (keys_iter != NULL){
                                while ((key = PyIter_Next(keys_iter)) != NULL) {
                                    if (PyUnicode_Check(key) && PyUnicode_GET_LENGTH(key) > 0) {
                                        PyObject *value, *item;
                                        const char *key_name = PyUnicode_AsUTF8(key); 
                                        value = PyDict_GetItem(dict, key);
                                        if (value != NULL) {
                                            char c[MAX_TYPE_REPR] = {0};
                                            _Type_Representation(value, c, class_repr, visited_obj);
                                            char * k = PyUnicode_AsUTF8(key);
                                            if(strlen(c) + strlen(dict_repr) + strlen(k) + 10 >= MAX_TYPE_REPR){
                                                ;
                                            }
                                            else{
                                                strcat(dict_repr, k);
                                                strcat(dict_repr, "$$");
                                                strcat(dict_repr, c);
                                                strcat(dict_repr, ",");
                                            }
                                        }
                                    }
                                    Py_DECREF(key);
                                }
                            }
                        }
                        Py_XDECREF(keys);
                        Py_XDECREF(keys_iter);
                        Py_XDECREF(dict);
                        if(strlen(repr) + 10 < MAX_TYPE_REPR){
                            strcat(dict_repr, "}");
                        }
                    }
                    if(strlen(class_name) + strlen(class_repr) + strlen(dict_repr) + 10 >= 10*MAX_TYPE_REPR){
                        ;
                    }
                    else{
                        strcat(class_repr, class_name);
                        strcat(class_repr, " <*> ");
                        strcat(class_repr, dict_repr);
                        strcat(class_repr, ",");
                    }
                }
                fprintf(fp, "%s: (%s)\n", base_class_name, class_repr);
                if(fp){
                    fclose(fp);
                }   
            }
        }
    }
    if(class_repr || BaseType(obj)){
        ;
    }
    else {
        
        
        PyObject * addr = PyLong_FromLong((size_t) obj);
        if(PySet_Contains(visited_obj, addr)){
            strcat(repr, " rec-ref");
            return;
        }
        PySet_Add(visited_obj, addr);
        if(strlen(repr) + 10 < MAX_TYPE_REPR){
            strcat(repr, " <*> {");
        }
        PyObject **dictptr = _PyObject_GetDictPtr(obj);
        if(dictptr && *dictptr){
            PyObject *key;
            PyObject *keys = NULL, *keys_iter = NULL;
            Py_INCREF(*dictptr);
            keys = PyDict_Keys(*dictptr);
            if (keys != NULL){
                keys_iter = PyObject_GetIter(keys);
                if (keys_iter != NULL){
                    while ((key = PyIter_Next(keys_iter)) != NULL) {
                        if (PyUnicode_Check(key) && PyUnicode_GET_LENGTH(key) > 0) {
                            PyObject *value, *item;
                            const char *key_name = PyUnicode_AsUTF8(key); 
                            value = PyDict_GetItem(*dictptr, key);
                            if (value != NULL) {
                                char c[MAX_TYPE_REPR] = {0};
                                _Type_Representation(value, c, class_repr, visited_obj);
                                char * k = PyUnicode_AsUTF8(key);
                                if(strlen(c) + strlen(repr) + strlen(k) + 10 >= MAX_TYPE_REPR){
                                    ;
                                }
                                else{
                                    strcat(repr, k);
                                    strcat(repr, "$$");
                                    strcat(repr, c);
                                    strcat(repr, ",");
                                }
                            }
                        }
                        Py_DECREF(key);
                    }
                }
            }
            Py_XDECREF(keys);
            Py_XDECREF(keys_iter);
            Py_XDECREF(*dictptr);
        }
        if(strlen(repr) + 10 < MAX_TYPE_REPR){
            strcat(repr, "}");
        }
    }
}
static void Type_Representation(PyObject * obj, char * repr, int class_repr){
    PyObject *visited_obj = NULL;
    if(class_repr == 0){
        // set_new would call collect_generations, which would segment fault in some circumstances (e.g., initing a list). 
        visited_obj = PySet_New(NULL);
    }
    
    _Type_Representation(obj, repr, class_repr, visited_obj);
    Py_XDECREF(visited_obj);

}
// Fast inlined version of PyType_IS_GC()
#define _PyType_IS_GC(t) _PyType_HasFeature((t), Py_TPFLAGS_HAVE_GC)

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_OBJECT_H */
