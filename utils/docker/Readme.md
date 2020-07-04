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
