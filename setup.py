from setuptools import setup, find_packages

setup(
    name='audioviz',
    packages=['audioviz'],
    version='0.1.0',
    description='An user-friendly music information retrieval tools interfacing with Google Colab',

    url='https://github.com/TrangDuLam/NTHU_Music_AI_Tools',
    author='ayTrang',
    author_email='andrew.chuang@gapp.nthu.edu.tw',
    license='MIT',
    zip_safe=False,
    keywords=['Music Information Retrieval', "Academia Sinica", "NTHU"],

    install_requires = ["numpy", 
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
    
    python_requires='>=3.9',

)
