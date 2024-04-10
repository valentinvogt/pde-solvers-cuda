#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include <convolution.hpp>
#include <convolve_sigma_add_f.hpp>
#include <helpers.hpp>
#include <io/netcdf_writer.hpp>
#include <neumann_bc.hpp>
#include <periodic_bc.hpp>
#include <string>
#include <zisa/io/file_manipulation.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/io/hierarchical_file.hpp>
#include <zisa/io/hierarchical_reader.hpp>
#include <zisa/io/netcdf_file.hpp>
#include <zisa/io/netcdf_serial_writer.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_traits.hpp>
#include <zisa/memory/device_type.hpp>
#include <zisa/memory/memory_location.hpp>
#include <zisa/memory/shape.hpp>

enum BoundaryCondition { Dirichlet, Neumann, Periodic };

template <typename Scalar> class PDEBase {
public:
  // note here that Nx and Ny denote the size INSIDE the boundary WITHOUT the
  // boundary so that the total size is Nx + 2 * Ny + 2
  PDEBase(unsigned Nx, unsigned Ny, const zisa::device_type memory_location,
          BoundaryCondition bc, Scalar dx, Scalar dy)
      : data_(zisa::shape_t<2>(Nx + 2, Ny + 2), memory_location),
        bc_neumann_values_(zisa::shape_t<2>(Nx + 2, Ny + 2), memory_location),
        sigma_values_(zisa::shape_t<2>(2 * Nx + 1, Ny + 1), memory_location),
        memory_location_(memory_location), bc_(bc), dx_(dx), dy_(dy) {}

  virtual void apply(Scalar dt) = 0;

  // apply timesteps and save snapshots at times T/n_snapshots
  // note that for this we sometimes have to change the timestep
  void apply_with_snapshots(Scalar T, unsigned int n_timesteps,
                            unsigned int n_snapshots,
                            NetCDFPDEWriter<Scalar> &writer) {

    Scalar dt = T / n_timesteps;
    Scalar time = 0.;
    unsigned int snapshot_counter = 0;
    Scalar dsnapshots = T / (n_snapshots - 1);
    // save initial data
    writer.save_snapshot(0, snapshot_counter, data_.const_view());
    snapshot_counter++;
    for (unsigned int i = 0; i < n_timesteps; ++i) {
      if (time + dt >= dsnapshots * snapshot_counter) {
        Scalar dt_new = dsnapshots * snapshot_counter - time;
        apply(dt_new);
        writer.save_snapshot(0, snapshot_counter, data_.const_view());
        apply(dt - dt_new);
        snapshot_counter++;
      } else {
        apply(dt);
      }
      time += dt;
    }

    if (snapshot_counter < n_snapshots) {
      // total Time doesn't reach T due to numerical errors. Add more timesteps
      Scalar dt_new = T - time;
      apply(dt_new);
      writer.save_snapshot(0, snapshot_counter, data_.const_view());
    }
  }

  // for testing, does this work if on gpu?
  zisa::array_const_view<Scalar, 2> get_data() { return data_.const_view(); }

  zisa::array_const_view<Scalar, 2> get_sigma() {
    return sigma_values_.const_view();
  }

  zisa::array_const_view<Scalar, 2> get_bc() {
    return bc_neumann_values_.const_view();
  }

  BoundaryCondition get_bc_type() { return bc_; }

  // for testing/debugging
  void print() {
    std::cout << "data has size x: " << data_.shape(0)
              << ", y: " << data_.shape(1) << std::endl;

    std::cout << "data:" << std::endl;
    print_matrix(data_.const_view());
    // do not print bc and sigma
    // return;
    std::cout << "bc values:" << std::endl;
    print_matrix(bc_neumann_values_.const_view());
    std::cout << "sigma values:" << std::endl;
    print_matrix(sigma_values_.const_view());
  }

protected:
  // apply boundary conditions
  // for cuda implementation, this should probably be done in the same step as
  // applying the convolution to avoid copying data back and forth
  void add_bc() {
    if (bc_ == BoundaryCondition::Dirichlet) {
      // do nothing as long as data on boundary does not change
      // dirichlet_bc(data_.view(), bc_neumann_values_.const_view());
    } else if (bc_ == BoundaryCondition::Neumann) {
      // TODO: change dt
      Scalar dt = 0.1;
      neumann_bc(data_.view(), bc_neumann_values_.const_view(), dt);
    } else if (bc_ == BoundaryCondition::Periodic) {
      periodic_bc(data_.view());
    } else {
      std::cout << "boundary condition not implemented yet!" << std::endl;
    }
  }

  zisa::array<Scalar, 2> data_;
  zisa::array<Scalar, 2> bc_neumann_values_;
  zisa::array<Scalar, 2> sigma_values_;

  const BoundaryCondition bc_;
  const zisa::device_type memory_location_;

  const Scalar dx_;
  const Scalar dy_;
  bool ready_ = false;
};

#endif // PDE_BASE_HPP_
