from setuptools import setup, find_packages
from typing import List


hypen_e = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path,"r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if hypen_e in requirements:
            requirements.remove(hypen_e)

    return requirements


setup(
    name='concrete_compressive_strength_prediction',
    version='0.0.1',    
    description='Build an end to end machine learning project that predicts the compressive strength of concrete using mix proportions, water-cement ratio, curing conditions, and age of the sample.',
    author='Pavan Ram Kalyan Boorla',
    author_email='pavanramkalyanboorla@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
              
)
