// #include <cxxtest/TestSuite.h>
#include <mfem.hpp>
#include <admfem.hpp>
// #include <miniapps/autodiff/admfem.hpp>
#include <fstream>
#include <iostream>
#include <optparser.hpp>
#include <device.hpp>
#include <forall.hpp>
// #include "general/optparser.hpp"
// #include "general/device.hpp"
// #include "IGXException.hpp"
// #include <catch.hpp>
// #include "tests/unit/catch.hpp"
// #include <Eigen/Core>
// #include <FnLogging.hpp>

void printVector( const mfem::Vector& vec, const char* str )
{
	std::cout << "\n" << str << " = " << std::endl;
	for( int i = 0; i < vec.Size(); ++i ){
		std::cout << vec[ i ] << ", ";
	}
	std::cout<< std::endl;
}
void printMatrix( const mfem::DenseMatrix& mat, const char* str )
{
	std::cout << "\n"  << str << " = " << std::endl;
	for( int i = 0; i < mat.Height(); ++i ){
		for( int j = 0; j < mat.Width(); ++j ){
			std::cout << mat( i, j ) << ", ";
		}
		std::cout << std::endl;
	}
}
void printArray( const mfem::Array<int>& arr, const char* str )
{
	std::cout << "\n" << str << " = " << std::endl;
	for( int i = 0; i < arr.Size(); ++i ){
		std::cout << arr[ i ] << ", ";
	}
	std::cout<< std::endl;
}

void assertVector( const mfem::Vector& vec1, const mfem::Vector& vec2, const double tol = 1e-12 )
{
	std::cout << "\nChecking the vector size..." <<  std::endl;
	MFEM_ASSERT( vec1.Size() == vec2.Size(), "vec1.Size() == vec2.Size()" );

	std::cout << "Checking the vector values..." <<  std::endl;
	for( int i = 0; i < vec1.Size(); ++i ){
		MFEM_ASSERT( std::abs(vec1[ i ] - vec2[ i ] ) < tol, "std::abs(vec1[ i ] - vec2[ i ] ) < tol" );
	}
}
void assertMatrix( const mfem::DenseMatrix& mat1, const mfem::DenseMatrix& mat2, const double tol = 1e-12 )
{
	std::cout << "\nChecking the matrix size..." <<  std::endl;
	MFEM_ASSERT( mat1.Height() == mat2.Height(), "mat1.Height() == mat2.Height()" );
	MFEM_ASSERT( mat1.Width() == mat2.Width(), "mat1.Width() == mat2.Width()" );

	std::cout << "Checking the matrix values..." <<  std::endl;
	for( int i = 0; i < mat1.Height(); ++i ){
		for( int j = 0; j < mat1.Width(); ++j ){
			MFEM_ASSERT( std::abs( mat1( i, j ) - mat2( i, j ) ) < tol, "std::abs( mat1( i, j ) - mat2( i, j ) ) < tol" );
		}
	}
}
void assertTranspose( const mfem::DenseMatrix& mat1, const mfem::DenseMatrix& mat2, const double tol = 1e-12 )
{
	std::cout << "\nChecking the matrix size..." <<  std::endl;
	MFEM_ASSERT( mat1.Height() == mat2.Width(), "mat1.Height() == mat2.Width()" );
	MFEM_ASSERT( mat1.Width() == mat2.Height(), "mat1.Width() == mat2.Height()" );

	std::cout << "Checking the matrix values..." <<  std::endl;
	for( int i = 0; i < mat1.Height(); ++i ){
		for( int j = 0; j < mat1.Width(); ++j ){
			MFEM_ASSERT( std::abs( mat1( i, j ) - mat2( j, i ) ) < tol, "std::abs( mat1( i, j ) - mat2( j, i ) ) < tol" );
		}
	}
}

namespace mfem
{
	template< typename TDataType, typename TParamVector, typename TStateVector,
           	  int residual_size, int state_size, int param_size >
	class NavierStokesResidual
	{
		public:
		// Residual functor
		void operator()( TParamVector& param, TStateVector& state_vector, TStateVector& residual )
		{
			// u: velocity
			// p: pressure
			// The following state vector is for a general case.
			// state_vector = [ ux, dux/dx, dux/dy, dux/dz, uy, duy/dx, duy/dy, duy/dz, uz, duz/dx, duz/dy, duz/dz, p, dp/dx, dp/dy, dp/dz ]
			// Each residual component is associated with each state_vector component in the same order.

			auto density = param[0];
			auto mu = param[1];

			const int dim = 3;
			const int n_per_dof = dim + 1;  // dof itself and its gradient entries

			// residual = 0.0;
         for( int i = 0; i < residual.Size(); ++i ) residual[i] = 0.0;
			// pure velocity term
			for( int i = 0; i < dim; ++i ){
				for( int j = 0; j < dim; ++j ){
					// density * grad(u) \cdot u
					residual[ n_per_dof * i ] += density * state_vector[ n_per_dof * i + j + 1 ] * state_vector[ n_per_dof * j ];
					// mu * grad(u)
					residual[ n_per_dof * i + j + 1 ] += mu * state_vector[ n_per_dof * i + j + 1 ];
				}
			}
			// velocity test - pressure trial
			for( int i = 0; i < dim; ++i )
				residual[ n_per_dof*i + i + 1 ] -= state_vector[ n_per_dof * dim ];
			// pressure test - velocity trial
			for( int i = 0; i < dim; ++i )
				residual[ n_per_dof * dim ] -= state_vector[ n_per_dof*i + i + 1 ];
		}	
	};
	
	template< typename TDataType, typename TParamVector, typename TStateVector,
           	  int residual_size, int state_size, int param_size >
	class ThermoNavierStokesResidual
	{
		public:
		// Residual functor
		void operator()( TParamVector& param, TStateVector& state_vector, TStateVector& residual )
		{
			// u: velocity
			// p: pressure
			// T: temperature
			// The following state vector is for a general case.
			// state_vector = [ ux, dux/dx, dux/dy, dux/dz, uy, duy/dx, duy/dy, duy/dz, uz, duz/dx, duz/dy, duz/dz, p, dp/dx, dp/dy, dp/dz, T, dT/dx, dT/dy, dT/dz ]
			// Each residual component is associated with each state_vector component in the same order.

			auto density = param[0];
			auto mu = param[1];
			auto cp = param[2];
			auto thermal_conductivity = param[3]; // thermal conductivity

			const int dim = 3;
			const int n_per_dof = dim + 1;  // dof itself and its gradient entries

			// residual = 0.0;
         for( int i = 0; i < residual.Size(); ++i ) residual[i] = 0.0;
			// pure velocity term
			for( int i = 0; i < dim; ++i ){
				for( int j = 0; j < dim; ++j ){
					// density * grad(u) \cdot u
					residual[ n_per_dof * i ] += density * state_vector[ n_per_dof * i + j + 1 ] * state_vector[ n_per_dof * j ];
					// mu * grad(u)
					residual[ n_per_dof * i + j + 1 ] += mu * state_vector[ n_per_dof * i + j + 1 ];
				}
			}
			// velocity test - pressure trial
			for( int i = 0; i < dim; ++i )
				residual[ n_per_dof*i + i + 1 ] -= state_vector[ n_per_dof * dim ];
			// pressure test - velocity trial
			for( int i = 0; i < dim; ++i )
				residual[ n_per_dof * dim ] -= state_vector[ n_per_dof*i + i + 1 ];
			// temperature test			
			// density * cp * u \cdot grad(T)
			for( int i = 0; i < dim; ++i )
				residual[ n_per_dof * ( dim + 1 ) ] += density * cp * state_vector[ n_per_dof * i ] * state_vector[ n_per_dof * ( dim + 1 ) + i + 1 ];
			// k * grad(T)
			for( int i = 0; i < dim; ++i )
				residual[ n_per_dof * ( dim + 1 ) + i + 1 ] = thermal_conductivity * state_vector[ n_per_dof * ( dim + 1 ) + i + 1 ];
		}	
	};
	

	// Penalty EBC
	template< typename TDataType, typename TParamVector, typename TStateVector,
           	  int residual_size, int state_size, int param_size >
	class ThermoNavierStokesEBCResidual
	{
		public:
		// Residual functor
		void operator()( TParamVector& param, TStateVector& state_vector, TStateVector& residual )
		{
			// u: velocity
			// p: pressure
			// T: temperature
			// The following state vector is for a general case.
			// state_vector = [ ux, uy, uz, p, T ]
			// Each residual component is associated with each state_vector component in the same order.
			// param entries 0 to 4: boundary values for state_vector 0 to 4.
			// param entries 5 to 9: penalty parameters for state_vector 0 to 4.

			const int dim = 3;

			for( int i = 0; i < dim + 2 ; ++i )
				residual[ i ] = param[ dim + i ] * ( state_vector[ i ] - param[ i ] );
		}	
	};

	// Integrator on a single element for multi-variable problems.
	// Takes multiple function spaces, each of which is associated with each variable.
	// Works for an arbitrary number of independent variables (e.g, velocity/pressure or velocity/pressure/temperature)
	// The number of variables must match the number of function spaces to pass in.
	// The mfem::BlockNonlinearForm class family (not mfem::BlockNonlinearFormIntegrator) iterates over elements calling this class and 
	//   sends the computed element residuals and tangents to a block sparse matrix, which is used by a nonlinear solver such as mfem::NewtonSolver.
	// This can take any RESIDUALCLASS class as a template parameter, as long as the class contains a functor of residual.
	//   One restriction on the residual functor is that it must be written in terms of the variables and their first gradients for generalization.
	//   The state variable order is [ V1, V2, V3, ... ]
	//   Vi = [ vi, dvi/dx, dvi/dy, dvi/dz ] if vi is a scalar field in 3D. vi denotes the i-th independent variable.
	//   Vi = [ vix, dvix/dx, dvix/dy, dvix/dz, viy, dviy/dx, dviy/dy, dviy/dz, viz, dviz/dx, dviz/dy, dviz/dz ] if vi is a vector.
	//   For example, see mfem::ThermoNavierStokesResidual.
	//   The residual entry order is consistent with the state entry order.
	template < template<typename,typename,typename,int,int,int> class RESIDUALCLASS, int DIM, int RESIDUALSIZE, int STATESIZE, int PARAMSIZE >
	class AutoDiffBasedBlockIntegrator : public BlockNonlinearFormIntegrator
	{
		protected:
		int mDim = DIM;  // space dimension
		int mNumParams = PARAMSIZE;
		mfem::QVectorFuncAutoDiff< RESIDUALCLASS, RESIDUALSIZE, STATESIZE, PARAMSIZE > mResidualFunc;

		private:
		Vector mParamVec;
		const int mNumSpaces;
		Array< int > mVDims;
		std::vector< Vector > mShape;
		std::vector< DenseMatrix > mDshapeIso;
		std::vector< DenseMatrix > mDshapeXyz;
		std::vector< DenseMatrix > mB;
		std::vector< DenseMatrix > mBCluster;
		std::vector< Vector > mPerSpaceState;
		std::vector< Array<int> > mPerSpaceStateIds;
		Vector mStateVector;
		Vector mResidualVector;
		DenseMatrix mResidualGrad;
		std::vector< Vector > mPerSpaceResidual;

		public:
		AutoDiffBasedBlockIntegrator( double *data_, const Array< int >& vsizes ): 
			mParamVec( data_, mNumParams ),
			mNumSpaces( vsizes.Size() ),
			mVDims( vsizes ),
			mShape( mNumSpaces ),
			mDshapeIso( mNumSpaces ),
			mDshapeXyz( mNumSpaces ),
			mB( mNumSpaces ),
			mBCluster( mNumSpaces ),
			mPerSpaceState( mNumSpaces ),
			mPerSpaceStateIds( mNumSpaces ),
			mStateVector( STATESIZE ),
			mResidualVector( RESIDUALSIZE ),
			mResidualGrad( RESIDUALSIZE, STATESIZE ),
			mPerSpaceResidual( mNumSpaces )
		{}

		virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                       ElementTransformation &trans,
                                       const Array<const Vector *> &elfun,
                                       const Array<Vector *> &elvect)
		{
			const int num_spaces = el.Size();
			// if( num_spaces != mNumSpaces ) COUT_THROW_IGXEXCEPTION( "The number of function spaces is not correct." );
			Array<int> ndofs(num_spaces);
			Array<int> vdims = mVDims;
			Array<int> orders(num_spaces);
			Array<int> offsets_dof(num_spaces + 1);
			Array<int> offsets_state(num_spaces + 1);
			offsets_dof[0] = 0;
			offsets_state[0] = 0;
			for( int space = 0; space < num_spaces; ++space ){
				ndofs[space] = el[space]->GetDof();
				orders[space] = el[space]->GetOrder();
				offsets_dof[space+1] = vdims[space] * ndofs[space];
				offsets_state[space+1] = vdims[space] * ( mDim + 1 );
			}
			offsets_dof.PartialSum();
			offsets_state.PartialSum();

			// if( offsets_state[ offsets_state.Size() - 1 ] != STATESIZE ) COUT_THROW_IGXEXCEPTION( "The state size is incorrect." );

			for( int space = 0; space < num_spaces; ++space ){
				const int state_size = offsets_state[space+1] - offsets_state[space];
				mPerSpaceState[space].SetSize( state_size );
				mPerSpaceStateIds[space].SetSize( state_size );
				std::iota( mPerSpaceStateIds[space].begin(), mPerSpaceStateIds[space].end(), offsets_state[space] );
			}

			const int order = orders.Max();
			const IntegrationRule& ir( IntRules.Get( el[0]->GetGeomType(), order + 3 ) );

			// Set shape function containers for each space
			for( int space = 0; space < num_spaces; ++space ){
				const int ntdof = vdims[space] * ndofs[space];  // num total element dofs for this space.
				mShape[space].SetSize( ndofs[space] );
				mDshapeIso[space].SetSize( ndofs[space], mDim );
				mDshapeXyz[space].SetSize( ndofs[space], mDim );
				mB[space].SetSize( ndofs[space], mDim + 1 );
				mBCluster[space].SetSize( vdims[space]*ndofs[space], vdims[space]*( mDim + 1 ) );
				mBCluster[space] = 0.0;
				
				mPerSpaceResidual[space].SetSize( ntdof );

				elvect[space]->SetSize( ntdof );
				*(elvect[space]) = 0.0;

			}

			double weight;

			for( int i = 0; i < ir.GetNPoints(); ++i )
			{
				const IntegrationPoint& ip = ir.IntPoint( i );
				trans.SetIntPoint( &ip );
				weight = trans.Weight();
				weight = ip.weight * weight;

				for( int space = 0; space < num_spaces; ++space ){
					const FiniteElement& elem = *(el[space]);
					Vector& shape = mShape[space];
					DenseMatrix& shape_iso = mDshapeIso[space];
					DenseMatrix& shape_xyz = mDshapeXyz[space];
					DenseMatrix& B_mat = mB[space];

					elem.CalcShape( ip, shape );
					elem.CalcDShape( ip, shape_iso );
					Mult( shape_iso, trans.InverseJacobian(), shape_xyz );
					B_mat.SetCol( 0, shape );
					for( int jj = 0; jj < mDim; ++jj ){
						B_mat.SetCol( jj+1, shape_xyz.GetColumn( jj ) );
					}
					DenseMatrix& B_cluster = mBCluster[space];
					if( vdims[space] > 1 ){
						for( int jj = 0; jj < vdims[space]; ++jj ){
							B_cluster.SetSubMatrix( jj * ndofs[space], jj * ( mDim + 1 ), B_mat );
						}
					} else{
						B_cluster = B_mat;
					}

					Vector& state_vec = mPerSpaceState[space];

					B_cluster.MultTranspose( *(elfun[space]), state_vec );
					mStateVector.SetSubVector( mPerSpaceStateIds[space], state_vec );
				}
				
				// residual
				mResidualFunc.VectorFunc( mParamVec, mStateVector, mResidualVector );
				// For each space, slice the residual vector and premultiply it by the shape function.
				for( int space = 0; space < num_spaces; ++space ){
					Vector& residual_vec = mPerSpaceState[space]; // reuse the container.
					mResidualVector.GetSubVector( mPerSpaceStateIds[space], residual_vec );
					mBCluster[space].Mult( residual_vec, mPerSpaceResidual[space] );
					elvect[space]->Add( weight, mPerSpaceResidual[space] );
				}

			}

		}

		virtual void AssembleElementGrad(const Array<const FiniteElement *> &el,
                                       ElementTransformation &trans,
                                       const Array<const Vector *> &elfun,
                                       const Array2D<DenseMatrix *> &elmats)
		{
			const int num_spaces = el.Size();
			// if( num_spaces != mNumSpaces ) COUT_THROW_IGXEXCEPTION( "The number of function spaces is not correct." );
			Array<int> ndofs(num_spaces);
			Array<int> vdims = mVDims;
			Array<int> orders(num_spaces);
			Array<int> offsets_dof(num_spaces + 1);
			Array<int> offsets_state(num_spaces + 1);
			offsets_dof[0] = 0;
			offsets_state[0] = 0;
			for( int space = 0; space < num_spaces; ++space ){
				ndofs[space] = el[space]->GetDof();
				orders[space] = el[space]->GetOrder();
				offsets_dof[space+1] = vdims[space] * ndofs[space];
				offsets_state[space+1] = vdims[space] * ( mDim + 1 );
			}
			offsets_dof.PartialSum();
			offsets_state.PartialSum();

			// if( offsets_state[ offsets_state.Size() - 1 ] != STATESIZE ) COUT_THROW_IGXEXCEPTION( "The state size is incorrect." );

			for( int space = 0; space < num_spaces; ++space ){
				const int state_size = offsets_state[space+1] - offsets_state[space];
				mPerSpaceState[space].SetSize( state_size );
				mPerSpaceStateIds[space].SetSize( state_size );
				std::iota( mPerSpaceStateIds[space].begin(), mPerSpaceStateIds[space].end(), offsets_state[space] );
			}

			const int order = orders.Max();
			const IntegrationRule& ir( IntRules.Get( el[0]->GetGeomType(), order + 3 ) );

			// Set shape function containers for each space
			for( int space = 0; space < num_spaces; ++space ){
				const int ntdof = vdims[space] * ndofs[space];  // num total element dofs for this space.
				mShape[space].SetSize( ndofs[space] );
				mDshapeIso[space].SetSize( ndofs[space], mDim );
				mDshapeXyz[space].SetSize( ndofs[space], mDim );
				mB[space].SetSize( ndofs[space], mDim + 1 );
				mBCluster[space].SetSize( vdims[space]*ndofs[space], vdims[space]*( mDim + 1 ) );
				mBCluster[space] = 0.0;
				
				for( int space2 = 0; space2 < num_spaces; ++space2 ){
					const int ntdof2 = vdims[space2] * ndofs[space2];
					elmats(space,space2)->SetSize( ntdof, ntdof2 );
					*(elmats(space,space2)) = 0.0;
				}

			}

			double weight;

			for( int i = 0; i < ir.GetNPoints(); ++i )
			{
				const IntegrationPoint& ip = ir.IntPoint( i );
				trans.SetIntPoint( &ip );
				weight = trans.Weight();
				weight = ip.weight * weight;

				for( int space = 0; space < num_spaces; ++space ){
					const FiniteElement& elem = *(el[space]);
					Vector& shape = mShape[space];
					DenseMatrix& shape_iso = mDshapeIso[space];
					DenseMatrix& shape_xyz = mDshapeXyz[space];
					DenseMatrix& B_mat = mB[space];

					elem.CalcShape( ip, shape );
					elem.CalcDShape( ip, shape_iso );
					Mult( shape_iso, trans.InverseJacobian(), shape_xyz );
					B_mat.SetCol( 0, shape );
					for( int jj = 0; jj < mDim; ++jj ){
						B_mat.SetCol( jj+1, shape_xyz.GetColumn( jj ) );
					}
					DenseMatrix& B_cluster = mBCluster[space];
					if( vdims[space] > 1 ){
						for( int jj = 0; jj < vdims[space]; ++jj ){
							B_cluster.SetSubMatrix( jj * ndofs[space], jj * ( mDim + 1 ), B_mat );
						}
					} else{
						B_cluster = B_mat;
					}


					Vector& state_vec = mPerSpaceState[space];

					B_cluster.MultTranspose( *(elfun[space]), state_vec );
					mStateVector.SetSubVector( mPerSpaceStateIds[space], state_vec );
				}
				
				// residual gradient
				mResidualFunc.Jacobian( mParamVec, mStateVector, mResidualGrad );
				// For each space, slice the residual vector and premultiply it by the shape function.
				for( int testspace = 0; testspace < num_spaces; ++testspace ){
					for( int trialspace = 0; trialspace < num_spaces; ++trialspace ){
						DenseMatrix sub_mat;
						mResidualGrad.GetSubMatrix( mPerSpaceStateIds[testspace], mPerSpaceStateIds[trialspace], sub_mat );
						DenseMatrix temp_mat( mBCluster[testspace].Height(), sub_mat.Width() );
						Mult( mBCluster[testspace], sub_mat, temp_mat );
						AddMult_a_ABt( weight, temp_mat, mBCluster[trialspace], *elmats( testspace, trialspace ) );
					}
				}

			}

		}
	};

	// Boundary integrator
	template < template<typename,typename,typename,int,int,int> class RESIDUALCLASS, int DIM, int RESIDUALSIZE, int STATESIZE, int PARAMSIZE >
	class AutoDiffBasedBlockBoundaryIntegrator : public BlockNonlinearFormIntegrator
	{
		protected:
		int mDim = DIM;  // space dimension
		int mNumParams = PARAMSIZE;
		mfem::QVectorFuncAutoDiff< RESIDUALCLASS, RESIDUALSIZE, STATESIZE, PARAMSIZE > mResidualFunc;

		private:
		Vector mParamVec;
		const int mNumSpaces;
		Array< int > mVDims;
		std::vector< Vector > mShape;
		std::vector< DenseMatrix > mN;
		std::vector< DenseMatrix > mNCluster;
		std::vector< Vector > mPerSpaceState;
		std::vector< Array<int> > mPerSpaceStateIds;
		Vector mStateVector;
		Vector mResidualVector;
		DenseMatrix mResidualGrad;
		std::vector< Vector > mPerSpaceResidual;
		// This function is user-defined.
		// It takes a physical coordinates and return a vector of parameters that used by the user-defined residual functor.
		std::function< void( const mfem::Vector&, mfem::Vector& ) > mParamFunc;
		Vector mCoordinates;

		public:
		AutoDiffBasedBlockBoundaryIntegrator( const Array< int >& vsizes,
									  const std::function< void( const mfem::Vector&, mfem::Vector& ) >& param_func ): 
			mParamVec( mNumParams ),
			mNumSpaces( vsizes.Size() ),
			mVDims( vsizes ),
			mShape( mNumSpaces ),
			mN( mNumSpaces ),
			mNCluster( mNumSpaces ),
			mPerSpaceState( mNumSpaces ),
			mPerSpaceStateIds( mNumSpaces ),
			mStateVector( STATESIZE ),
			mResidualVector( RESIDUALSIZE ),
			mResidualGrad( RESIDUALSIZE, STATESIZE ),
			mPerSpaceResidual( mNumSpaces ),
			mParamFunc( param_func ),
			mCoordinates( mDim )
		{}
		
		virtual void AssembleFaceVector(const Array<const FiniteElement *> &el,
										const Array<const FiniteElement *> &el2,
										FaceElementTransformations &trans,
										const Array<const Vector *> &elfun,
										const Array<Vector *> &elvect)
		{
			// The shape function vector will be constructed in a similar manner to the domain integrator.
			// The parameter vector will be constructed through a lambda function that will be a member of this class.
			//     - The reason to do this to allow flexible definition of the parameter
			//     - Can be similarly done for domain
			// For the block nonlinear form, el and el2 are identical, so only el is used here.
			
			const int num_spaces = el.Size();
			// if( num_spaces != mNumSpaces ) COUT_THROW_IGXEXCEPTION( "The number of function spaces is not correct." );
			Array<int> ndofs(num_spaces);
			Array<int> vdims = mVDims;
			Array<int> orders(num_spaces);
			Array<int> offsets_dof(num_spaces + 1);
			Array<int> offsets_state(num_spaces + 1);
			offsets_dof[0] = 0;
			offsets_state[0] = 0;
			for( int space = 0; space < num_spaces; ++space ){
				ndofs[space] = el[space]->GetDof();
				orders[space] = el[space]->GetOrder();
				offsets_dof[space+1] = vdims[space] * ndofs[space];
				offsets_state[space+1] = vdims[space];
			}
			offsets_dof.PartialSum();
			offsets_state.PartialSum();

			std::cout << "\033[38;5;214m" << "offsets_state.PartialSum()" << "\033[0m" << std::endl;

			// if( offsets_state[ offsets_state.Size() - 1 ] != STATESIZE ) COUT_THROW_IGXEXCEPTION( "The state size is incorrect." );

			for( int space = 0; space < num_spaces; ++space ){
				const int state_size = offsets_state[space+1] - offsets_state[space];
				mPerSpaceState[space].SetSize( state_size );
				mPerSpaceStateIds[space].SetSize( state_size );
				std::iota( mPerSpaceStateIds[space].begin(), mPerSpaceStateIds[space].end(), offsets_state[space] );
			}

			std::cout << "\033[38;5;214m" << "std::iota" << "\033[0m" << std::endl;

			const int order = orders.Max();
			const IntegrationRule& ir( IntRules.Get( el[0]->GetGeomType(), order + 3 ) );

			// Set shape function containers for each space
			for( int space = 0; space < num_spaces; ++space ){
				const int ntdof = vdims[space] * ndofs[space];  // num total element dofs for this space.
				mShape[space].SetSize( ndofs[space] );
				mN[space].SetSize( ndofs[space], 1 );
				mNCluster[space].SetSize( vdims[space]*ndofs[space], vdims[space] );
				mNCluster[space] = 0.0;
				
				mPerSpaceResidual[space].SetSize( ntdof );

				elvect[space]->SetSize( ntdof );
				*(elvect[space]) = 0.0;

			}

			std::cout << "\033[38;5;214m" << "*(elvect[space]) = 0.0" << "\033[0m" << std::endl;

			double weight;

			for( int i = 0; i < ir.GetNPoints(); ++i )
			{
				const IntegrationPoint& ip = ir.IntPoint( i );
				trans.SetIntPoint( &ip );
				weight = trans.Weight();
				weight = ip.weight * weight;

				for( int space = 0; space < num_spaces; ++space ){
					const FiniteElement& elem = *(el[space]);
					Vector& shape = mShape[space];
					DenseMatrix& N_mat = mN[space];

					elem.CalcShape( ip, shape );
					N_mat.SetCol( 0, shape );
					DenseMatrix& N_cluster = mNCluster[space];
					if( vdims[space] > 1 ){
						for( int jj = 0; jj < vdims[space]; ++jj ){
							N_cluster.SetSubMatrix( jj * ndofs[space], jj, N_mat );
						}
					} else{
						N_cluster = N_mat;
					}

					Vector& state_vec = mPerSpaceState[space];

					N_cluster.MultTranspose( *(elfun[space]), state_vec );
					mStateVector.SetSubVector( mPerSpaceStateIds[space], state_vec );
				}

				trans.Transform( ip, mCoordinates );
				mParamFunc( mCoordinates, mParamVec );
				
				// residual
				mResidualFunc.VectorFunc( mParamVec, mStateVector, mResidualVector );
				// For each space, slice the residual vector and premultiply it by the shape function.
				for( int space = 0; space < num_spaces; ++space ){
					Vector& residual_vec = mPerSpaceState[space]; // reuse the container.
					mResidualVector.GetSubVector( mPerSpaceStateIds[space], residual_vec );
					mNCluster[space].Mult( residual_vec, mPerSpaceResidual[space] );
					elvect[space]->Add( weight, mPerSpaceResidual[space] );
				}

			}

		}

		virtual void AssembleFaceGrad(const Array<const FiniteElement *>&el,
									const Array<const FiniteElement *>&el2,
									FaceElementTransformations &trans,
									const Array<const Vector *> &elfun,
									const Array2D<DenseMatrix *> &elmats)
		{
			const int num_spaces = el.Size();
			// if( num_spaces != mNumSpaces ) COUT_THROW_IGXEXCEPTION( "The number of function spaces is not correct." );
			Array<int> ndofs(num_spaces);
			Array<int> vdims = mVDims;
			Array<int> orders(num_spaces);
			Array<int> offsets_dof(num_spaces + 1);
			Array<int> offsets_state(num_spaces + 1);
			offsets_dof[0] = 0;
			offsets_state[0] = 0;
			for( int space = 0; space < num_spaces; ++space ){
				ndofs[space] = el[space]->GetDof();
				orders[space] = el[space]->GetOrder();
				offsets_dof[space+1] = vdims[space] * ndofs[space];
				offsets_state[space+1] = vdims[space];
			}
			offsets_dof.PartialSum();
			offsets_state.PartialSum();

			// if( offsets_state[ offsets_state.Size() - 1 ] != STATESIZE ) COUT_THROW_IGXEXCEPTION( "The state size is incorrect." );

			for( int space = 0; space < num_spaces; ++space ){
				const int state_size = offsets_state[space+1] - offsets_state[space];
				mPerSpaceState[space].SetSize( state_size );
				mPerSpaceStateIds[space].SetSize( state_size );
				std::iota( mPerSpaceStateIds[space].begin(), mPerSpaceStateIds[space].end(), offsets_state[space] );
			}

			const int order = orders.Max();
			const IntegrationRule& ir( IntRules.Get( el[0]->GetGeomType(), order + 3 ) );

			// Set shape function containers for each space
			for( int space = 0; space < num_spaces; ++space ){
				const int ntdof = vdims[space] * ndofs[space];  // num total element dofs for this space.
				mShape[space].SetSize( ndofs[space] );
				mN[space].SetSize( ndofs[space], 1 );
				mNCluster[space].SetSize( vdims[space]*ndofs[space], vdims[space] );
				mNCluster[space] = 0.0;
				
				for( int space2 = 0; space2 < num_spaces; ++space2 ){
					const int ntdof2 = vdims[space2] * ndofs[space2];
					elmats(space,space2)->SetSize( ntdof, ntdof2 );
					*(elmats(space,space2)) = 0.0;
				}

			}

			double weight;

			for( int i = 0; i < ir.GetNPoints(); ++i )
			{
				const IntegrationPoint& ip = ir.IntPoint( i );
				trans.SetIntPoint( &ip );
				weight = trans.Weight();
				weight = ip.weight * weight;

				for( int space = 0; space < num_spaces; ++space ){
					const FiniteElement& elem = *(el[space]);
					Vector& shape = mShape[space];
					DenseMatrix& N_mat = mN[space];

					elem.CalcShape( ip, shape );
					N_mat.SetCol( 0, shape );
					DenseMatrix& N_cluster = mNCluster[space];
					if( vdims[space] > 1 ){
						for( int jj = 0; jj < vdims[space]; ++jj ){
							N_cluster.SetSubMatrix( jj * ndofs[space], jj, N_mat );
						}
					} else{
						N_cluster = N_mat;
					}


					Vector& state_vec = mPerSpaceState[space];

					N_cluster.MultTranspose( *(elfun[space]), state_vec );
					mStateVector.SetSubVector( mPerSpaceStateIds[space], state_vec );
				}

				trans.Transform( ip, mCoordinates );
				mParamFunc( mCoordinates, mParamVec );
				
				// residual gradient
				mResidualFunc.Jacobian( mParamVec, mStateVector, mResidualGrad );
				// For each space, slice the residual vector and premultiply it by the shape function.
				for( int testspace = 0; testspace < num_spaces; ++testspace ){
					for( int trialspace = 0; trialspace < num_spaces; ++trialspace ){
						DenseMatrix sub_mat;
						mResidualGrad.GetSubMatrix( mPerSpaceStateIds[testspace], mPerSpaceStateIds[trialspace], sub_mat );
						DenseMatrix temp_mat( mNCluster[testspace].Height(), sub_mat.Width() );
						Mult( mNCluster[testspace], sub_mat, temp_mat );
						AddMult_a_ABt( weight, temp_mat, mNCluster[trialspace], *elmats( testspace, trialspace ) );
					}
				}

			}

		}
	};

	// Block nonlinear system solver
	// This uses Newton solver, GMRESSolver, and AutoDiffBasedBlockIntegrator.
	class BlockNonlinearSystemSolver
	{
		private:
		Mesh mMesh;
		Array< FiniteElementSpace* > mSpaces;
		BlockNonlinearForm *mNonlinearform = nullptr;
		NewtonSolver *mNonlinearSolver = nullptr;
		GMRESSolver *mLinearSolver = nullptr;
		// MINRESSolver *mLinearSolver = nullptr;
		
		// nonlinear solver parameters
		double mNLSolverRtol = 1e-6;
		double mNLSolverAtol = 1e-8;
		int mNLSolverIter = 10;
	
		// linear solver parameters
		double mLSolverRtol = 1e-4;
		double mLSolverAtol = 1e-5;
		int mLSolverIter = 5000;
	
		int mPrintLevel = 1;

		public:
		BlockNonlinearSystemSolver( const Mesh& mesh,
									const Array< FiniteElementSpace* >& spaces ):
					mMesh( mesh ),
					mSpaces( spaces )
		{
			mNonlinearform = new BlockNonlinearForm( mSpaces );
		}
	
		~BlockNonlinearSystemSolver()
		{
			deallocSolvers();
		}

		void solve( Vector& solution_coeff )
		{
			allocSolvers();
			std::cout << "\033[33m" << "allocSolvers done" << "\033[0m" << std::endl;
			Vector rhs_vec; // for now rhs is not set
			mNonlinearSolver->Mult( rhs_vec, solution_coeff );
			solution_coeff.HostRead();
		}

		void setNonlinearForm( const std::function< void( BlockNonlinearForm& form ) >& form_setup_func )
		{
			form_setup_func( *mNonlinearform );
			std::cout << "\033[33m" << "Set up nonlinear form" << "\033[0m" << std::endl;
		}

		private:
		void deallocSolvers()
		{
			if (mNonlinearform!=nullptr) { delete mNonlinearform;}
			if (mNonlinearSolver!=nullptr) { delete mNonlinearSolver;}
			if (mLinearSolver!=nullptr) { delete mLinearSolver;}
		}
		void allocSolvers()
		{
			deallocSolvers();

			// mfem::Array< int > vdims = getVariableDimensions();
			// mNonlinearform->AddDomainIntegrator( new AutoDiffBasedBlockIntegrator< RESIDUALCLASS, DIM, RESIDUALSIZE, STATESIZE, PARAMSIZE >( mParams.GetData(), vdims ) );
			// mNonlinearform->SetEssentialBC( bdr_attr_is_ess, rhs );

			mLinearSolver = new GMRESSolver();
			// mLinearSolver = new MINRESSolver();
			mLinearSolver->SetRelTol( mLSolverRtol );
			mLinearSolver->SetAbsTol( mLSolverAtol );
			mLinearSolver->SetMaxIter( mLSolverIter );
			mLinearSolver->SetPrintLevel( mPrintLevel );

			std::cout << "\033[33m" << "Set linear solver" << "\033[0m" << std::endl;

			mNonlinearSolver = new NewtonSolver();
			mNonlinearSolver->iterative_mode = true;
			mNonlinearSolver->SetSolver( *mLinearSolver );
			mNonlinearSolver->SetOperator( *mNonlinearform );
			mNonlinearSolver->SetRelTol( mNLSolverRtol );
			mNonlinearSolver->SetAbsTol( mNLSolverAtol );
			mNonlinearSolver->SetMaxIter( mNLSolverIter );
			mNonlinearSolver->SetPrintLevel( mPrintLevel );

			std::cout << "\033[33m" << "Set nonlinear solver" << "\033[0m" << std::endl;
		}
		mfem::Array< int > getVariableDimensions()
		{
			mfem::Array< int > vdims( mSpaces.Size() );
			for( int s = 0; s < mSpaces.Size(); ++s ) vdims[s] = mSpaces[s]->GetVDim();
			return vdims;
		}

	};

	// This solves a multi-variable nonlinear problem with all Dirichlet boundaries.
	// Takes the information about residual, mesh, variable dimension, function space orders, boundary conditions, physical parameters from outside.
	// No need to pass in anything about the tangent. The automatic differentiation does that.
	// Template parameters
	//   RESIDUALCLASS: The class that contains the residual functor of this problem. e.g. mfem::ThermoNavierStokesResidual
	//   DIM: space dimension. e.g. 3
	template< int DIM >
	class MultiVariableProblemSolver
	{
		private:
		mfem::Mesh *mMesh;
		int mNumSpaces;
		mfem::Array< int > mSpaceOrders;
		mfem::Array< int > mVariableDimensions;
		std::vector< std::string > mPropagationFiles;
		Device mDevice;

		MemoryType mMemoryType;

		mfem::Array< mfem::H1_FECollection* > mFECollections;
		mfem::Array< mfem::FiniteElementSpace* > mFESpaces;

		mfem::Array< int > mOffsets;
		int mTotalDofCounts;
		mfem::Vector mFuncCoeffs;
		// mfem::BlockVector mFuncCoeffs;

		std::vector< mfem::GridFunction > mNodalCoordinates;

		mfem::Array< mfem::Array< int >* > mBdrAttrIsEssential;
		
		mfem::BlockNonlinearSystemSolver *mBlockNLSystemSolver;

		public:
		MultiVariableProblemSolver( mfem::Mesh *mesh, 
								   const mfem::Array< int >& space_orders, 
								   const mfem::Array< int >& var_dims, 
								   std::vector< std::string >& prolongation_file_names,
								   const char *device_config = "cpu" ):
			mMesh( mesh ),
			mNumSpaces( var_dims.Size() ),
			mSpaceOrders( space_orders ),
			mVariableDimensions( var_dims ),
			mPropagationFiles( prolongation_file_names ),
			mDevice( device_config )
		{
			// if( mesh->Dimension() != DIM ) COUT_THROW_IGXEXCEPTION( "The mesh dimension is not equal to DIM" );
			// if( params.size() != PARAMSIZE ) COUT_THROW_IGXEXCEPTION( "The number of parameters passed in is different from PARAMSIZE." );
			// if( RESIDUALSIZE != STATESIZE ) COUT_THROW_IGXEXCEPTION( "RESIDUALSIZE must be equal to STATESIZE." );
			// if( mVariableDimensions.Sum()*(DIM+1) != STATESIZE ) COUT_THROW_IGXEXCEPTION( "STATESIZE mismatches the number of state variables calculated based on the variable dimensions and the space dimension." );

			std::cout << "\033[33m" << "AllDirichletProblemSolver construction starts" << "\033[0m" << std::endl;

			mDevice.Print();
			mMemoryType = mDevice.GetMemoryType();

			setFunctionSpace();
			std::cout << "\033[33m" << "setFunctionSpace() done" << "\033[0m" << std::endl;
			getOffsets();
			std::cout << "\033[33m" << "getOffsets done" << "\033[0m" << std::endl;
			setFunctionCoeffs();
			std::cout << "\033[33m" << "setFunctionCoeffs done" << "\033[0m" << std::endl;
			getNodalCoordinates();
			std::cout << "\033[33m" << "getNodalCoordinates done" << "\033[0m" << std::endl;
			mBlockNLSystemSolver = new mfem::BlockNonlinearSystemSolver( *mMesh, mFESpaces );
			std::cout << "\033[33m" << "Block nonlinear solver setup done" << "\033[0m" << std::endl;
		}

		void setNonlinearForm( const std::function< void( BlockNonlinearForm& form ) >& form_setup_func )
		{
			mBlockNLSystemSolver->setNonlinearForm( form_setup_func );
		}
		void solve()
		{		
			mfem::StopWatch timer;
			timer.Clear(); timer.Start();
			mBlockNLSystemSolver->solve( mFuncCoeffs );
			timer.Stop();
			std::cout << "The solution time is: " << timer.RealTime() << std::endl;
		}
		const mfem::Vector& functionCoefficients(){ return mFuncCoeffs; }
		const std::vector< mfem::GridFunction >& nodalCoordinates(){ return mNodalCoordinates; }
		const mfem::Array< int >& offset(){ return mOffsets; }

		private:
		void setFunctionSpace()
		{
			mFECollections.SetSize( mNumSpaces );
			mFESpaces.SetSize( mNumSpaces );
			for( int s = 0; s < mNumSpaces; ++s ){
				mFECollections[s] = new mfem::H1_FECollection( mSpaceOrders[s], DIM );
				std::cout << "\033[33m" << "H1_FECollection done" << "\033[0m" << std::endl;
				mFESpaces[s] = new mfem::FiniteElementSpace( mMesh, mFECollections[s], mVariableDimensions[s] );
				std::cout << "\033[33m" << "FiniteElementSpace done" << "\033[0m" << std::endl;
				// Set an identity prolongation/restriction matrices
				setProlongationAndRestriction( mFESpaces[s], mPropagationFiles[s] );
				std::cout << "\033[33m" << "setProlongationAndRestriction done" << "\033[0m" << std::endl;
			}
		}
		void setProlongationAndRestriction( FiniteElementSpace *fes, std::string& fname )
		{
			std::cout << "\033[33m" << "setProlongationAndRestriction starts" << "\033[0m" << std::endl;
			// // const int vdim = fes->GetTrueVSize();
			const int vdim = fes->GetVSize();  // temporary workaround to test the identity prolongation.
			std::cout << "\033[33m" << "vdim = " << vdim << "\033[0m" << std::endl;

			std::ifstream in_prolongation( fname );
			fes->LoadProlongation( in_prolongation );
			std::ifstream in_restriction( fname );
			fes->LoadRestriction( in_restriction );
		}
		void getOffsets()
		{
			mOffsets.SetSize( mNumSpaces + 1 );
			mOffsets[0] = 0;
			for( int space = 0; space < mNumSpaces; ++space ){
				mOffsets[ space + 1 ] = mFESpaces[ space ]->GetTrueVSize();
			}
			mOffsets.PartialSum();
			printArray(mOffsets, "mOffsets");
			mTotalDofCounts = mOffsets[ mOffsets.Size() - 1 ];
		}
		void setFunctionCoeffs()
		{
			mFuncCoeffs.SetSize( mTotalDofCounts, mMemoryType );
			mFuncCoeffs = 0.0;
		}
		void getNodalCoordinates()
		{
			mNodalCoordinates.resize( mNumSpaces );
			
			for( int space = 0; space < mNumSpaces; ++space ){
				mNodalCoordinates[ space ] = getNodalCoordinatesFromSpace( *mMesh, *mFECollections[ space ], DIM );
				mNodalCoordinates[ space ].HostRead();
			}
			// exit(EXIT_SUCCESS);
		}
		mfem::GridFunction getNodalCoordinatesFromSpace( mfem::Mesh& mesh, mfem::H1_FECollection& fec, const int dim )
		{
			mfem::FiniteElementSpace fespace_temp( &mesh, &fec, dim );
			mfem::GridFunction nodal_coord(&fespace_temp);
			mesh.GetNodes(nodal_coord);
			return nodal_coord;
		}
	};
}

std::vector< mfem::Vector > getExactTNS( const std::vector< mfem::GridFunction >& xyz, const int dim, const double density )
{
   std::vector< mfem::Vector > v(3);
   {
      const int s = 0;
      const int num_points = xyz[s].Size() / dim;
      v[s].SetSize(3*num_points); // velocity
      for( int i = 0; i < num_points; ++i ){
         const double x = xyz[s][ 0*num_points + i ];
         const double y = xyz[s][ 1*num_points + i ];
         const double z = xyz[s][ 2*num_points + i ];

         v[ s ][ 0*num_points + i ] = 2*x;
         v[ s ][ 1*num_points + i ] = -y;
         v[ s ][ 2*num_points + i ] = -z;
      }
   }
   {
      const int s = 1;
      const int num_points = xyz[s].Size() / dim;
      v[s].SetSize(1*num_points); // presesure
      for( int i = 0; i < num_points; ++i ){
         const double x = xyz[s][ 0*num_points + i ];
         const double y = xyz[s][ 1*num_points + i ];
         const double z = xyz[s][ 2*num_points + i ];

         v[ s ][ 0*num_points + i ] = -density * ( 2*x*x + .5*y*y + .5*z*z );
      }
   }
   {
      const int s = 2;
      const int num_points = xyz[s].Size() / dim;
      v[s].SetSize(1*num_points); // temperature
      for( int i = 0; i < num_points; ++i ){
         const double x = xyz[s][ 0*num_points + i ];
         const double y = xyz[s][ 1*num_points + i ];
         const double z = xyz[s][ 2*num_points + i ];

         v[ s ][ 0*num_points + i ] = x*y*z;
      }
   }

   return v;
}


int main(int argc, char *argv[])
{
	  std::cout << "\n\nTest_Patch_ThermoNS_AllDirichletProblemSolver" <<  std::endl;

      const char *device_config = "cpu";

      mfem::OptionsParser args(argc, argv);
      args.AddOption(&device_config, "-d", "--device",
                     "Device configuration string, see Device::Configure().");
      args.Parse();
      if (!args.Good())
      {
         args.PrintUsage(std::cout);
         return 1;
      }
      args.PrintOptions(std::cout);

      // // Enable hardware devices such as GPUs, and programming models such as
      // // CUDA, OCCA, RAJA and OpenMP based on command line options.
      // mfem::Device device(device_config);
      // device.Print();
      

		const int num_variables = 3; // velocity, pressure, temperature

		// Build mesh for [0,1]^3 cube.
		const int num_elem_1d = 10;
		mfem::Mesh *mesh = new mfem::Mesh(mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON ));
		std::cout << "\033[33m" << "mesh created" << "\033[0m" << std::endl;
		// mfem::NCMesh *ncmesh = new mfem::NCMesh( mesh );
		// std::cout << "\033[33m" << "ncmesh created" << "\033[0m" << std::endl;
		mesh->ncmesh = new mfem::NCMesh(mesh);
		std::cout << "\033[33m" << "mesh->ncmesh = ncmesh" << "\033[0m" << std::endl;

		// mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON );

		// problem parameters
		const double rho = 1.;
		const double mu = 2.;
		const double cp = 3.;
		const double k = 4.;
		std::vector< double > params({ rho, mu, cp, k });

		// variable dimensions
		constexpr int space_dim = 3;
		mfem::Array<int> var_dims( num_variables );
		var_dims[ 0 ] = space_dim;  // velocity
		var_dims[ 1 ] = 1;  // pressure
		var_dims[ 2 ] = 1;  // temperature

		// function space orders
		mfem::Array<int> orders( num_variables );
		orders[ 0 ] = 2;  // velocity
		orders[ 1 ] = 1;  // pressure
		orders[ 2 ] = 1;  // temperature

		// EBC penalty
		const double mesh_size = 1. / num_elem_1d;
		const double normalized_pen = 1e1;
		mfem::Array<double> pen_ebc( num_variables );
		pen_ebc[ 0 ] = normalized_pen * mu / mesh_size;
		pen_ebc[ 1 ] = - normalized_pen / mu;
		pen_ebc[ 2 ] = normalized_pen * k / mesh_size;

		// prolongation files
		std::vector< std::string > prolongation_file_names = { 	"/mnt/c/Users/jhbae/Coreform/work/mfem/test/matrix0.txt",
																"/mnt/c/Users/jhbae/Coreform/work/mfem/test/matrix1.txt",
																"/mnt/c/Users/jhbae/Coreform/work/mfem/test/matrix2.txt" };

		// set solver
		mfem::MultiVariableProblemSolver< space_dim >
			solver( mesh, orders, var_dims, prolongation_file_names, device_config );

		constexpr int residual_size = 20;
		constexpr int state_size = 20;
		constexpr int param_size = 4;
		solver.setNonlinearForm( 
			[ & ]( mfem::BlockNonlinearForm& form )
			{
				form.AddDomainIntegrator( 
					new mfem::AutoDiffBasedBlockIntegrator< mfem::ThermoNavierStokesResidual, space_dim, residual_size, state_size, param_size >( 
							params.data(), 
							var_dims ) );

				form.AddBdrFaceIntegrator( 
					new mfem::AutoDiffBasedBlockBoundaryIntegrator< mfem::ThermoNavierStokesEBCResidual, space_dim, 5, 5, 10 >( 
							var_dims,
							[ &pen_ebc, &rho ]( const mfem::Vector& coords, mfem::Vector& params )
							{
								const double x = coords[0];
								const double y = coords[1];
								const double z = coords[2];
								params[0] = 2*x;
								params[1] = -y;
								params[2] = -z;
								params[3] = -rho * ( 2*x*x + .5*y*y + .5*z*z );
								params[4] = x*y*z;
								params[5] = pen_ebc[0];
								params[6] = pen_ebc[0];
								params[7] = pen_ebc[0];
								params[8] = pen_ebc[1];
								params[9] = pen_ebc[2];
							} ) );
			} );

		std::cout << "\033[33m" << "AllDirichletProblemSolver set up" << "\033[0m" << std::endl;

		// Solve
		solver.solve();

		const mfem::Vector& solution_coeff_vec = solver.functionCoefficients();

		
		// Exact solution
		const std::vector< mfem::GridFunction >& nodal_coords = solver.nodalCoordinates();
		std::cout << "\033[33m" << "Extracted nodal_coords" << "\033[0m" << std::endl;
		// get the exact solutions
		std::vector< mfem::Vector > exact_sols = getExactTNS( nodal_coords, space_dim, params[0] );
		std::cout << "\033[33m" << "getExactTNS done" << "\033[0m" << std::endl;

		// printVector( solution_coeff_vec, "solution_coeff_vec" );

		// Assert
		const mfem::Array< int >& offsets = solver.offset();
		mfem::Vector exact_sol( offsets[offsets.Size()-1] );
		for( int s = 0; s < num_variables; ++s ){
			for( int dof = 0; dof < offsets[s+1]-offsets[s]; ++dof ){
				exact_sol[ offsets[s] + dof ] = exact_sols[ s ][ dof ];
			}
		}
		mfem::Vector error(solution_coeff_vec.Size());
		for( int i = 0; i < solution_coeff_vec.Size(); ++i ) error[i] = solution_coeff_vec[i] - exact_sol[i];
		const double norm = error.Norml2();
		const double norm_tot = exact_sol.Norml2();
		const double norm_rel = norm / norm_tot / ( num_elem_1d*num_elem_1d*num_elem_1d );
		std::cout << "\nrelative error norm = " << norm_rel << std::endl;

		MFEM_ASSERT( norm_rel < 1e-6, "norm_rel < 1e-6" );
      return 0;
}
