#ifndef __viz_3_h__
#define __viz_3_h__

#include "stdint.h"

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"

#ifdef __cplusplus
#include "legion/legion_c_util.h"
#endif

// C API bindings

#ifdef __cplusplus
extern "C" {
#endif

void viz_tasks_register(void);

typedef struct main_launcher_t { void *impl; } main_launcher_t;

main_launcher_t main_launcher_create(legion_predicate_t pred /* = legion_predicate_true() */, legion_mapper_id_t id /* = 0 */, legion_mapping_tag_id_t tag /* = 0 */);

void main_launcher_destroy(main_launcher_t launcher);

void main_launcher_set_enable_inlining(main_launcher_t launcher, bool enable_inlining);

legion_future_t main_launcher_execute(legion_runtime_t runtime, legion_context_t context, main_launcher_t launcher);

#ifdef __cplusplus
}
#endif

// C++ API bindings

#ifdef __cplusplus

class main_launcher {
public:
   main_launcher(Legion::Predicate pred = Legion::Predicate::TRUE_PRED, Legion::MapperID id = 0, Legion::MappingTagID tag = 0) {
    predicate = new Legion::Predicate(pred);
    launcher = main_launcher_create(Legion::CObjectWrapper::wrap(predicate), id, tag);
  }
   ~main_launcher() {
    main_launcher_destroy(launcher);
    delete predicate;
  }
  void set_enable_inlining(bool enable_inlining) {
    main_launcher_set_enable_inlining(launcher, enable_inlining);
  }
  Legion::Future execute(Legion::Runtime *runtime, Legion::Context ctx) {
    legion_runtime_t c_runtime = Legion::CObjectWrapper::wrap(runtime);
    Legion::CContext c_ctx(ctx);
    legion_context_t c_context = Legion::CObjectWrapper::wrap(&c_ctx);
    legion_future_t c_future = main_launcher_execute(c_runtime, c_context, launcher);
    Legion::Future future = *Legion::CObjectWrapper::unwrap(c_future);
    legion_future_destroy(c_future);
    return future;
  }
private:
  Legion::Predicate *predicate;
  main_launcher_t launcher;
};

#endif

#endif // __viz_3_h__
