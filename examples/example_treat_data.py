############################################################################
#    Code to analyse the data
#
#    Copyright (C) 2022  Gaetan Facchinetti
#    gaetan.facchinetti@ulb.be
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>
############################################################################


import py21cmfast.dm_dtb_tools as db_tools
import example_lightcone_analysis as lightcone_analysis

# Define the database location
database_location = "/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/test_database"
cache_location = "/scratch/ulb/physth_fi/gfacchin/output_exo21cmFAST/"

db_manager_approx = db_tools.ApproxDepDatabase(path = database_location, cache_path = cache_location)
db_manager_DH     = db_tools.DHDatabase(path = database_location, cache_path = cache_location)

index_0 = db_manager_approx.search(lifetime=1e+26, approx_params=[[0.3], [0.4, 0.2]], mDM=1e+8)
index_1 = db_manager_DH.search(lifetime=0, primary='none', mDM=0)

# We can charge and remake the analysis for the lightcones in index_0 with
for ind in index_0:
    path = db_manager_approx.path_brightness_temp
    lightcone = lightcone_analysis.LightCone.read(fname= path + str(ind) + '/Lightcone/Lightcone.h5')
    lightcone_analysis.make_analysis(path + str(ind), lightcone, n_psbins=20, nchunks=40)
