#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:06:18 2021

@author: saldern 
all fenics felics functions

"""
import numpy as np

# azimuthal expansion
def ExpandForAzimuthalAverage(coordinates,n_cuts):
    ## This function extends the fenics grid to three dimensions to prepare for interpolation in 3D grids
    size=np.shape(coordinates)[0]
    b=np.zeros((size,3))
    b[:,0]=coordinates[:,0]
    b[:,1]=coordinates[:,1]
    coordinates=b
     
    temp_coordinates=np.zeros(np.shape(coordinates))
    coordinates_out=np.zeros((0,3))
    for angle in np.linspace(2*np.pi/n_cuts,2*np.pi,n_cuts):
        temp_coordinates[:,0]=coordinates[:,0]
        temp_coordinates[:,1]=coordinates[:,1]*np.cos(angle)
        temp_coordinates[:,2]=coordinates[:,1]*np.sin(angle)
        coordinates_out=np.concatenate((coordinates_out,temp_coordinates))
    return coordinates_out

# azimuthal average
def ContractFromAximuthalAverage(coordinates,n_cuts,vec_in):
    ## This function contracts the expanded interpolated data from the expaned grid (see function ExpandForZaimuthalAverage) back
    ## to the 2D base grid
    length=np.shape(coordinates)[0]
    vec_out=np.zeros(length)
    for i in range(n_cuts):
        vec_out=vec_out+vec_in[i*length:i*length+length]
    vec_out=vec_out/n_cuts  
    return vec_out

# from dolfin import *
# from fenics import *

# write data with coordinates to a Fenics function
def write2FenicsFun(V,mesh_fenics,data,coo2):
	## Define Variables
	v = Function(V)
	dofmap = V.dofmap()
	nvertices = mesh_fenics.ufl_cell().num_vertices()
	
	# Set up a vertex_2_dof list
	indices = [dofmap.tabulate_entity_dofs(0, i)[0] for i in range(nvertices)]
	vertex_2_dof = dict()
	[vertex_2_dof.update(dict(vd for vd in zip(cell.entities(0), dofmap.cell_dofs(cell.index())[indices]))) for cell in cells(mesh_fenics)]
	
	# Get the vertex coordinates
	X = mesh_fenics.coordinates()
	
	for ind in range(0,len(data)):
		# Set the vertex coordinate you want to modify
		rcoord, xcoord = coo2[ind,1], coo2[ind,0] # here the orientation is changing again
		
		# Find the matching vertex (if it exists)
		vertex_idx = np.where((X == (rcoord,xcoord)).all(axis = 1))[0] 
		if not vertex_idx:
			print('No matching vertex for point with coordinates:')
			print(rcoord, xcoord)
			if sum((rcoord, xcoord)==mesh_fenics.coordinates()[0,:])==2:
				vertex_idx = vertex_idx[0]
				dof_idx = vertex_2_dof[vertex_idx]
				v.vector()[dof_idx] = data[ind]
				print('problem with coordinate')
				print(mesh_fenics.coordinates()[0,:])
				print(' is fixed!')
		else:
			vertex_idx = vertex_idx[0]
			dof_idx = vertex_2_dof[vertex_idx]
			#print(dof_idx)
			v.vector()[dof_idx] = data[ind]
	return v

# compute eddy-viscosity from Reynolds-Stresses (Thomas' function)
def nutFromBoussinesq(MeanFlowDict,R,FEMSpaces):
# it takes the Boussinesq ansatz an rearranges the equation such that nu_t is computed
# the six different nu_t resulting from this ansatz is accounted for by a least square ansatz and already included in the underneath implementation
	
	  
	# to avoid division by zero, set R(R=0)=NaN
	# this results in NaN later on in the results which will be set to the molecular viscosity
	R[R==0] = np.nan
	
	# HARD_CODED: all derivatives are performed in cylindrical coordinates
	# derivatives are smoothened to avoid noise due to derivative of nearest interpolated data !!!!!'''''''' ############
	# also do not need all intermediate results such as rstij and dui_dj a FEMspace in MeanflowDict, however it is handy to observe them later in paraview
	MeanFlowDict['dux_dx'] = Function(FEMSpaces)
	MeanFlowDict['dux_dx'].vector()[:] = project(MeanFlowDict['ux'].dx(0),FEMSpaces).vector()[:]
#	MeanFlowDict['dux_dx'] = smoothFieldWithKernel(MeanFlowDict['dux_dx'],FEMSpaces,15)
	
	MeanFlowDict['dux_dr'] = Function(FEMSpaces)
	MeanFlowDict['dux_dr'].vector()[:] = project(MeanFlowDict['ux'].dx(1),FEMSpaces).vector()[:]
#	MeanFlowDict['dux_dr'] = smoothFieldWithKernel(MeanFlowDict['dux_dr'],FEMSpaces,15)
	
	MeanFlowDict['dux_dt'] = Function(FEMSpaces)
	MeanFlowDict['dux_dt'].vector()[:] = MeanFlowDict['ux'].vector()[:]*0
#	MeanFlowDict['dux_dt'] = smoothFieldWithKernel(MeanFlowDict['dux_dt'],FEMSpaces,15)
	
	MeanFlowDict['dur_dx'] = Function(FEMSpaces)
	MeanFlowDict['dur_dx'].vector()[:] = project(MeanFlowDict['ur'].dx(0),FEMSpaces).vector()[:]
#	MeanFlowDict['dur_dx'] = smoothFieldWithKernel(MeanFlowDict['dur_dx'],FEMSpaces,15)
	
	MeanFlowDict['dur_dr'] = Function(FEMSpaces)
	MeanFlowDict['dur_dr'].vector()[:] = project(MeanFlowDict['ur'].dx(1),FEMSpaces).vector()[:]
#	MeanFlowDict['dur_dr'] = smoothFieldWithKernel(MeanFlowDict['dur_dr'],FEMSpaces,15)
	
	MeanFlowDict['dut_dt'] = Function(FEMSpaces)
	tmp = MeanFlowDict['ur'].vector()[:]/R
	tmp2 = project(MeanFlowDict['ur'].dx(1),FEMSpaces).vector()[:]
	idx = np.isnan(R)
#	tmp[idx] = tmp2[idx]
	tmp[idx] = 0
	MeanFlowDict['dut_dt'].vector()[:] = tmp 
#	MeanFlowDict['dut_dt'] = smoothFieldWithKernel(MeanFlowDict['dut_dt'],FEMSpaces,15)
	
	
	MeanFlowDict['dut_dx'] = Function(FEMSpaces)
	MeanFlowDict['dut_dx'].vector()[:] = project(MeanFlowDict['ut'].dx(0),FEMSpaces).vector()[:]
#	MeanFlowDict['dut_dx'] = smoothFieldWithKernel(MeanFlowDict['dut_dx'],FEMSpaces,15)
	
	MeanFlowDict['dut_dr'] = Function(FEMSpaces)
	MeanFlowDict['dut_dr'].vector()[:] = project(MeanFlowDict['ut'].dx(1),FEMSpaces).vector()[:]
#	MeanFlowDict['dut_dr'] = smoothFieldWithKernel(MeanFlowDict['dut_dr'],FEMSpaces,15)
	
	MeanFlowDict['dur_dt'] = Function(FEMSpaces)
	tmp = -MeanFlowDict['ut'].vector()[:]/R
	tmp2 = project(MeanFlowDict['ut'].dx(1),FEMSpaces).vector()[:]
	idx = np.isnan(R)
	tmp[idx] = tmp2[idx]
	MeanFlowDict['dur_dt'].vector()[:] = tmp 
#	MeanFlowDict['dur_dt'] = smoothFieldWithKernel(MeanFlowDict['dur_dt'],FEMSpaces,15)
	
	MeanFlowDict['TKE']=Function(FEMSpaces)
	MeanFlowDict['TKE'].vector()[:] = 0.5*(MeanFlowDict['rstxx'].vector()[:] + MeanFlowDict['rstrr'].vector()[:] + MeanFlowDict['rsttt'].vector()[:])
	 
	numerator = (-MeanFlowDict['rstxx'].vector()[:]+2.0/3.0*MeanFlowDict['TKE'].vector()[:])\
	*(MeanFlowDict['dux_dx'].vector()[:]+MeanFlowDict['dux_dx'].vector()[:])\
	+(-MeanFlowDict['rstxr'].vector()[:])*(MeanFlowDict['dux_dr'].vector()[:]+MeanFlowDict['dur_dx'].vector()[:])\
	+(-MeanFlowDict['rstxt'].vector()[:])*(MeanFlowDict['dux_dt'].vector()[:]+MeanFlowDict['dut_dx'].vector()[:])\
	+(-MeanFlowDict['rstxr'].vector()[:])*(MeanFlowDict['dur_dx'].vector()[:]+MeanFlowDict['dux_dr'].vector()[:])\
	+(-MeanFlowDict['rstrr'].vector()[:]+2.0/3.0*MeanFlowDict['TKE'].vector()[:])\
	*(MeanFlowDict['dur_dr'].vector()[:]+MeanFlowDict['dur_dr'].vector()[:])\
	+(-MeanFlowDict['rstrt'].vector()[:])*(MeanFlowDict['dur_dt'].vector()[:]+MeanFlowDict['dut_dr'].vector()[:])\
	+(-MeanFlowDict['rstxt'].vector()[:])*(MeanFlowDict['dut_dx'].vector()[:]+MeanFlowDict['dux_dt'].vector()[:])\
	+(-MeanFlowDict['rstrt'].vector()[:])*(MeanFlowDict['dut_dr'].vector()[:]+MeanFlowDict['dur_dt'].vector()[:])\
	+(-MeanFlowDict['rsttt'].vector()[:]+2.0/3.0*MeanFlowDict['TKE'].vector()[:])\
	*(MeanFlowDict['dut_dt'].vector()[:]+MeanFlowDict['dut_dt'].vector()[:])
	 
	denominator=+(MeanFlowDict['dux_dx'].vector()[:]+MeanFlowDict['dux_dx'].vector()[:])**2\
	+(MeanFlowDict['dux_dr'].vector()[:]+MeanFlowDict['dur_dx'].vector()[:])**2\
	+(MeanFlowDict['dux_dt'].vector()[:]+MeanFlowDict['dut_dx'].vector()[:])**2\
	+(MeanFlowDict['dur_dx'].vector()[:]+MeanFlowDict['dux_dr'].vector()[:])**2\
	+(MeanFlowDict['dur_dr'].vector()[:]+MeanFlowDict['dur_dr'].vector()[:])**2\
	+(MeanFlowDict['dur_dt'].vector()[:]+MeanFlowDict['dut_dr'].vector()[:])**2\
	+(MeanFlowDict['dut_dx'].vector()[:]+MeanFlowDict['dux_dt'].vector()[:])**2\
	+(MeanFlowDict['dut_dr'].vector()[:]+MeanFlowDict['dur_dt'].vector()[:])**2\
	+(MeanFlowDict['dut_dt'].vector()[:]+MeanFlowDict['dut_dt'].vector()[:])**2
	 
	MeanFlowDict['nuturb']=Function(FEMSpaces)
	MeanFlowDict['nuturb'].vector()[:] = numerator / denominator
	#MeanFlowDict['nuturb'] = FenicsInterpolate(MeanFlowDict['nu_t'], FunctionSpace(FEMSpaces.mesh(), 'P', 1))
	tmp = abs(MeanFlowDict['nuturb'].vector()[:])
#	tmp = MeanFlowDict['nuturb'].vector()[:]
#	tmp[tmp<0] = 0
	MeanFlowDict['nuturb'].vector()[:] = tmp
	del MeanFlowDict['ux']
	del MeanFlowDict['ur']
	return MeanFlowDict





















