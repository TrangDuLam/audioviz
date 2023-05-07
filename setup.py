from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='audioviz',
    packages=['audioviz'],
    version='0.1.7',
    
    description='An user-friendly music information retrieval tools interfacing with Google Colab',
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    url='https://github.com/TrangDuLam/audioviz',
    author='ayTrang',
    author_email='andrew.chuang@gapp.nthu.edu.tw',
    license='MIT',
    zip_safe=False,
    keywords=['Music Information Retrieval', "Academia Sinica", "NTHU"],

    install_requires = ["numpy", 
                        "matplotlib",
                        "pandas",
                        "scipy", 
                        "librosa", 
                        "madmom", 
                        "libfmp", 
                        "plotly",
                        "soundfile",
                        ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    
    python_requires='>=3.9, !=3.10',

)
