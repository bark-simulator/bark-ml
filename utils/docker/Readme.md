# Building and Uploading Images

## Docker
`docker login`
`docker build -t barksim/bark-ml .`
`docker push barksim/bark-ml:latest`
`docker system prune -a`

For diadem:
`docker build -t barksim/bark-ml-diadem .`
`docker push barksim/bark-ml-diadem:latest`


## Singularity
`sudo singularity build bark_ml.img Singularity`
`bash upload_image.sh hart`

## Cluster
Mount drive:
`sudo mount -t glusterfs -o acl fortiss-8gpu:/data /mnt/glusterdata`

also ssh key:
ssh-keygen -t rsa
ssh-copy-id -i ~/.ssh/id_<user>_cluster <user>@8gpu


# Work in progress
On macOS we can only perform a pull
singularity pull bark_ml.img docker://barksim/bark-ml:latest
singularity exec -B $PWD:/bark-ml bark_ml.img /bin/bash

docker run -it -v $PWD:/bark-ml barksim/bark-ml:latest /bin/bash
docker run -it -v $PWD:/bark-ml barksim/bark-ml:latest bazel run //examples:tfa_gnn
virtualenv -p python3 ./bark_ml/python_wrapper/venv --system-site-packages
. ./bark_ml/python_wrapper/venv/bin/activate


-- setup --
1. navigate to vagrant folder and run `vagrant up`
2. run `vagrant ssh`
3. run `cd /vagrant`

-- cluster --
4. run `singularity exec -B $PWD:/bark-ml bark_ml.img /bin/bash`
5. run  `cd /bark-ml/bark-ml`
6. run `virtualenv -p python3 ./bark_ml/python_wrapper/venv --system-site-packages`
7. run `. ./bark_ml/python_wrapper/venv/bin/activate`
8. eg run `bazel test //...`

-- run_bazel.sh --
#!bin/bash
cd /bark-ml/bark-ml
virtualenv -p python3 ./bark_ml/python_wrapper/venv --system-site-packages
bazel run //examples:tfa_gnn -- $@

-- command to be executed in sbatch --
singularity exec -B $PWD:/bark-ml bark_ml.img /bin/bash /bark-ml/run_bazel.sh
