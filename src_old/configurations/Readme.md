# Mount
sudo mkdir -p /mnt/glusterdata
sudo mount -t glusterfs -o acl fortiss-8gpu:/data /mnt/glusterdata


# Deploy
rsync ./configurations/run.sh  8gpu:/mnt/glusterdata/home/run.sh -a --copy-links -v -z -P
bash configurations/deploy.sh highway hart experiment_1

# Run
ssh 8gpu
sbatch run.sh experiment_01
sattach 11060.0
