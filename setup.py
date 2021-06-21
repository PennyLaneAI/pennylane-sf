# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
import sys
import os
from setuptools import setup

with open("pennylane_sf/_version.py") as f:
	version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = [
    "strawberryfields>=0.15",
    "pennylane>=0.15"
]

info = {
    'name': 'PennyLane-SF',
    'version': version,
    'maintainer': 'Xanadu Inc.',
    'maintainer_email': 'nathan@xanadu.ai',
    'url': 'https://github.com/XanaduAI/pennylane-sf',
    'license': 'Apache License 2.0',
    'packages': [
                    'pennylane_sf'
                ],
    'entry_points': {
        'pennylane.plugins': [
            'strawberryfields.remote = pennylane_sf:StrawberryFieldsRemote',
            'strawberryfields.fock = pennylane_sf:StrawberryFieldsFock',
            'strawberryfields.gaussian = pennylane_sf:StrawberryFieldsGaussian',
            'strawberryfields.gbs = pennylane_sf:StrawberryFieldsGBS',
            'strawberryfields.tf = pennylane_sf.tf:StrawberryFieldsTF'
            ],
        },
    'description': 'Open source library for continuous-variable quantum computation',
    'long_description': open('README.rst').read(),
    'provides': ["pennylane_sf"],
    'install_requires': requirements,
    # 'extras_require': extra_requirements,
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
