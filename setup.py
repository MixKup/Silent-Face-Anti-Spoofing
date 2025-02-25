from setuptools import setup, find_packages
def readme():
    with open('README.md') as f:
        return f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name='mkantispoofing',
    packages=find_packages(),
    url='https://github.com/MixKup/mkantispoofing.git',
    description='This is a description for antispoof',
    long_description=open('README.md').read(),
    install_requires=required,
    include_package_data=True,
)
