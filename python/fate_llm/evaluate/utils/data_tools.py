#
#  Copyright 2024 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


def download_data(data_dir, data_url, is_tar=True):
    import os
    import requests
    import tarfile
    import io

    # Create data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download data
    try:
        response = requests.get(data_url)
        if response.status_code == 200:
            if is_tar:
                # extract tar file and write to data_dir
                with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
                    for member in tar.getmembers():
                        # check if member is a file
                        if member.isreg():
                            member.name = os.path.join(data_dir, os.path.basename(member.name))
                            tar.extract(member)
            else:
                # write to data_dir
                with open(os.path.join(data_dir, os.path.basename(data_url)), 'wb') as f:
                    f.write(response.content)
            return True
        else:
            print(f"Error downloading file: {response.status_code}")
            return False

    except Exception as e:
        print(f"Error downloading file: {e}")
    return False
