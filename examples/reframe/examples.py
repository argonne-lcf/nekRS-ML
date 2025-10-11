import reframe as rfm
import reframe.utility.sanity as sn
from nekrs import NekRSCase, NekRSTest

class NekRSKershawCase(NekRSCase):

  case_name = 'kershaw'
  case_root = '/lus/flare/projects/Aurora_AT/nekRS/cases/kershaw'

  def __init__(self, case_number):
    super().__init__(name=self.case_name, directory=f'{self.case_root}/{self.case_name}_{case_number:04}')

@rfm.simple_test
class NekRSKershawTest(NekRSTest):

  case_number=variable(int, value=1)
  maximum_walltime = '00:40:00'

  valid_case_numbers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

  case_constraints = { 
    1   : {'min_nodes':1   , 'max_nodes':1   },
    2   : {'min_nodes':1   , 'max_nodes':2   },
    4   : {'min_nodes':1   , 'max_nodes':4   },
    8   : {'min_nodes':1   , 'max_nodes':8   },
    16  : {'min_nodes':2   , 'max_nodes':16  },
    32  : {'min_nodes':4   , 'max_nodes':32  }, 
    64  : {'min_nodes':8   , 'max_nodes':64  }, 
    128 : {'min_nodes':16  , 'max_nodes':128 }, 
    256 : {'min_nodes':32  , 'max_nodes':256 }, 
    512 : {'min_nodes':64  , 'max_nodes':512 }, 
    1024: {'min_nodes':128 , 'max_nodes':1024}, 
    2048: {'min_nodes':256 , 'max_nodes':2048}, 
    4096: {'min_nodes':512 , 'max_nodes':4096}, 
    8192: {'min_nodes':1024, 'max_nodes':8192}
  }

  def __init__(self):
    super().__init__(nekrs_case=NekRSKershawCase(self.case_number))

  @run_after('init')
  def validate_case_number(self):
    if self.case_number not in self.valid_case_numbers:
      self.skip(f'kershaw_{self.case_number} not valid a valid case. "case_number" be one of {self.valid_case_numbers}.')

  # Filter out tests which cannot run on the current node count
  @run_before('setup')
  def check_constraints(self):
    min_nodes = self.case_constraints[self.case_number]['min_nodes']
    max_nodes = self.case_constraints[self.case_number]['max_nodes']
    self.skip_if(self.num_nodes < min_nodes, f'This test requires at least {min_nodes} nodes.')
    self.skip_if(self.num_nodes > max_nodes, f'This should run on at most {max_nodes} nodes.')

  @run_after('setup')
  def set_run_parameters(self):
    self.set_walltime(self.maximum_walltime)

  # Match "flops/rank" at start of line to avoid matching output during setup.
  @performance_function('flops/s/rank', perf_key='BPS5')
  def bps5_performance(self):
    return sn.extractsingle(r'^flops/rank: (\S+)', self.stdout, 1, float, 0)

  @performance_function('flops/s/rank', perf_key='BP5')
  def bp5_performance(self):
    return sn.extractsingle(r'^flops/rank: (\S+)', self.stdout, 1, float, 1)

  @performance_function('flops/s/rank',perf_key='BP6')
  def bp6_performance(self):
    return sn.extractsingle(r'^flops/rank: (\S+)', self.stdout, 1, float, 2)

  @performance_function('flops/s/rank',perf_key='BP6PCG')
  def bp6pcg_performance(self):
    return sn.extractsingle(r'^flops/rank: (\S+)', self.stdout, 1, float, 3)
