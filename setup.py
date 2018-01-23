from distutils.core import setup

print('DOES NOT WORK! STILL A WORK IN PROGRESS')
setup(name='Bolt',
      version='1.0',
      packages=['bolt',
                'bolt.lib.linear',
                'bolt.lib.nonlinear'
               ]
     )