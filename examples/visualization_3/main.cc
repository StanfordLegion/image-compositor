#include <mpi.h>
#include "legion.h"
#include "viz_3.h"
#include "visualization_3_mapper.h"
#include "render.h"

using namespace Legion;
MPILegionHandshake handshake;

int main(int argc, char **argv)
{
#if defined(GASNET_CONDUIT_MPI) || defined(REALM_USE_MPI)
  // The GASNet MPI conduit and/or the Realm MPI network layer
  // require that MPI be initialized for multiple threads
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  // If you fail this assertion, then your version of MPI
  // does not support calls from multiple threads and you
  // cannot use the GASNet MPI conduit
  if (provided < MPI_THREAD_MULTIPLE)
    printf("ERROR: Your implementation of MPI does not support "
           "MPI_THREAD_MULTIPLE which is required for use of the "
           "GASNet MPI conduit or the Realm MPI network layer "
           "with the Legion-MPI Interop!\n");
  assert(provided == MPI_THREAD_MULTIPLE);
#else
  // Perform MPI start-up like normal for most GASNet conduits
  MPI_Init(&argc, &argv);
#endif
  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Runtime::configure_MPI_interoperability(rank);

  Runtime::set_top_level_task_id(TID_TOP_LEVEL_TASK);

  viz_tasks_register();
  register_mappers();

  handshake = Runtime::create_handshake(true/*MPI initial control*/,
                                        1/*MPI participants*/,
                                        1/*Legion participants*/);

  Runtime::start(argc, argv, true);

  handshake.mpi_handoff_to_legion();
  handshake.mpi_wait_on_legion();

  Runtime::wait_for_shutdown();
#ifndef GASNET_CONDUIT_MPI
  // Then finalize MPI like normal
  // Exception for the MPI conduit which does its own finalization
  MPI_Finalize();
#endif
  return 0;
}
