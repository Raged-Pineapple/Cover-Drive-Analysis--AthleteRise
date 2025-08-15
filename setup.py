from setuptools import setup, find_packages

setup(
    name="cricket-cover-drive-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "numpy",
        "pandas",
        "plotly",
        "python-dotenv",
        "google-generativeai",
        "mediapipe",
        "opencv-python",
        "scipy"
    ],
    python_requires='>=3.8',
    package_data={
        '': ['*.json', '*.yaml'],
    },
    include_package_data=True,
)
