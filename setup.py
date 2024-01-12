from setuptools import setup, find_packages

setup(
    name = 'brypy',
    packages = find_packages(),
    version = '0.1',
    author = 'Bryce Wedig',
    author_email = 'b.t.wedig@wustl.edu',
    url = 'https://github.com/bryce-wedig/brypy',
    license='MIT',
    install_requires=['numpy', 'matplotlib', 'astropy', 'scipy', 'pandas', 'tqdm']
)
