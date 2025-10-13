# Demonstrated features of nekRS-ML examples

## nekRS-ML examples

|                              | Dist-GNN | SR-GNN | Offline | Online w/ SmartSim | Online w/ ADIOS2 | p-coarsening |
|------------------------------|----------|--------|---------|--------------------|------------------|--------------|
| tgv_gnn_offline              |   x      |        |  x      |                    |                  |              |
| tgv_gnn_offline_fine_mesh    |   x      |        |  x      |                    |                  |       x      |
| tgv_gnn_traj_offline         |   x      |        |  x      |                    |                  |              |
| turbChannel_wallModel_ML     |          |        |         |        x           |                  |              |
| tgv_gnn_online               |   x      |        |         |        x           |                  |              |
| tgv_gnn_traj_online          |   x      |        |         |        x           |                  |              |
| tgv_gnn_traj_online_adios    |   x      |        |         |                    |        x         |              |
| shooting_workflow_smartredis |   x      |        |         |        x           |                  |              |
| shooting_workflow_adios      |   x      |        |         |                    |        x         |              |


## Plain nekRS examples

|                          | GAB | KTC | LMA | TPF | MVC | HMI | EDN | CHT | HIT |
|--------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| iofld                    |     |     |     |  x  |     |     |     |     |     |
| in-situ viz              |     |     |     |  x  |     |     |     |     |     |
| lowMach                  |     |     |  x  |     |  x  |     |     |     |     |
| varying p0th             |     |     |     |     |  x  |     |     |     |     |
| inflow recycling         |     |     |     |  x  |     |     |     |     |     |
| user source term         |  x  |  x  |  x  |  x  |  x  |     |     |     |  x  |
| implicit linear source   |     |     |     |     |     |     |     |     |  x  |
| scalar transport         |  x  |  x  |  x  |     |  x  |     |     |  x  |     |
| variable props           |  x  |  x  |  x  |     |  x  |     |     |     |     |
| Dirichlet BC             |  x  |     |  x  |     |     |  x  |  x  |  x  |     |
| flux (Neumann) BC        |  x  |     |     |     |     |     |     |  x  |     |
| traction BC              |  x  |     |     |     |     |     |     |     |     |
| sym BC                   |     |  x  |     |     |  x  |  x  |     |     |     |
| user BC data (usrwrk)    |  x  |     |     |     |     |     |     |     |     |
| turbulent outflow (Dong) |     |     |     |     |     |  x  |     |     |     |
| variable dt              |  x  |     |     |  x  |     |     |     |     |     |
| OIFS/subcycling          |  x  |     |     |  x  |     |     |  x  |     |     |
| Lagrangian particles     |     |     |     |     |     |  x  |     |     |     |
| point interpolation      |  x  |  x  |     |     |     |  x  |     |     |     |
| extract line (hpts)      |     |  x  |     |     |     |     |     |     |     |
| runtime averages         |  x  |     |     |     |     |     |     |     |     |
| planar average           |  x  |     |     |     |     |     |     |     |     |
| conjugate heat transfer  |     |     |     |     |     |     |     |  x  |     |
| RANS (k-tau)             |     |  x  |     |     |     |     |     |     |     |
| surfaceIntegral          |     |  x  |     |     |     |     |     |     |     |
| aero forces              |     |  x  |     |     |     |     |     |     |     |
| mesh manipulation        |  x  |     |     |     |     |     |     |     |     |
| moving mesh (ALE)        |     |     |     |     |  x  |  x  |     |     |     |
| constant flow rate       |     |  x  |     |     |     |     |     |     |     |
| overset grids (neknek)   |     |     |     |     |     |     |  x  |     |     |
| par casedata             |  x  |     |     |  x  |  x  |  x  |  x  |     |     |
| nek data exchange        |  x  |     |     |     |     |     |     |     |     |
| hpf-RT                   |  x  |     |     |  x  |     |  x  |     |     |     |
| avm (scalar)             |     |     |     |  x  |     |     |     |     |     |
| nek useric               |     |     |  x  |     |     |     |     |     |     |
| predictor-corrector iter |     |     |     |     |     |     |  x  |     |     |
| usrchk postprocessing    |     |     |  x  |     |     |     |     |     |     |
| opSEM                    |     |     |     |     |     |     |     |     |  x  |
| Q-criterion              |     |     |     |  x  |     |     |     |     |  x  |
| user checkpoint variable |     |     |     |  x  |     |     |     |     |     |
| user output              |     |     |     |  x  |     |     |     |     |     |

### Ledgend
`Dist-GNN`: [Distributed-GNN](https://ieeexplore.ieee.org/abstract/document/10820662) for global field modeling.

`SR-GNN`: [Super-Resolution-GNN]((https://www.sciencedirect.com/science/article/abs/pii/S0045782525003445)) for local super-resolution of coarse fields.

`GAB`: gabls1

`KTC`: ktauChannel

`LMA`: lowMach

`TPF`: turbPipe

`MVC`: mv_cyl

`HMI`: hemi

`EDN`: eddyNekNek                

`CHT`: conj_ht               

`HIT`: hit                
