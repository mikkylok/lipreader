# Lip Reader
This is the branch for deployment code. Any push will automatically trigger Github Action CICD to containerize the repo, push the image to ECR, and deploy it on EC2.
For development code, please refer to dev branch.

## Directory

- config/: app's central configuration

- model/: model files

- uploaded_videos/: two example videos

- utils/: 
  - face.py: get lip points from an image
  - video.py: get lip points from a video
  - predict.py: load model weights and predict

- app.py: starting script for streamlit demo

- Dockerfile: the docker file during containerization

- .github/workflows/: the deployment file for github actions: containerize, push the image to ECR, deploy on EC2


## Environment Requirement

Python >= 3.9


## Local Development Setup
1. Open a terminal and navigate to your project folder.

`$ cd lipreader`

2. In your terminal, type:

`$ python -m venv .venv`

3. A folder named ".venv" will appear in your project. This directory is where your virtual environment and its dependencies are installed.


4. In your terminal, activate your environment with one of the following commands, depending on your operating system.

`$ source .venv/bin/activate`

5. Download necessary packages

`$ pip install -r requirements. txt`

6. Run your Streamlit app.

`$ python -m streamlit run app.py`

7. To stop the Streamlit server, press `Ctrl+C` in the terminal.


8. When you're done using this environment, return to your normal shell by typing:

`$ deactivate`

## Local Docker Setup

1. Generate requirements, (pipreqs only generate requirements for the current project)

`$ pip install pipreqs`

`$ pipreqs . --ignore ".venv" `

2. Build image

`$ docker build -t your-image-name:tag .`

3. Run image

`$ docker run -d --name your-container-name -p 8501:8501 -v ~/.aws:/root/.aws your-image-name:latest`

4. Visit localhost:8501 in the browser

5. Cleanup work: stop container

`$ docker stop test-container`

6. Cleanup work: delete container

`$ docker rm test-container`

7. Cleanup work: delete image

`$ docker rmi <image_name>:<tag> `

## Demo Deployment steps
1. Apply for an IAM user with `Opensearchfullaccess` and `ECRfullaccess`


2. Create an EC2 instance and generate a key pair for SSH 
- Please meet model's basic hardware requirement
- Please choose a pytorch version with cuda and pytorch preinstalled

3. Create an ECR registry repo to store images
   
During configuration, please make the image is immutable!!

4. Try SSH to the server (Follow this tutorial: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html)

(1) `$ chmod 400 key-pair-name.pem`

(2) `$ ssh -i /path/key-pair-name.pem instance-user-name@instance-public-dns-name`


5. SSH to your machine to make initial configurations.

(1) Install docker (Follow this tutorial: https://medium.com/@srijaanaparthy/step-by-step-guide-to-install-docker-on-amazon-linux-machine-in-aws-a690bf44b5fe)

(2) Add your user to the Docker group to run Docker commands without 'sudo': 

`$ sudo usermod -a -G docker ec2-user`
   
After adding the user to the Docker group, simply disconnect and reconnect to SSH to refresh the group memberships.
   
(3) Configure AWS credentials on the server: `$ aws configure`, Input access key, secret key of the IAM user

(4) Log on to ECR on the server: please replace `123456789012.dkr.ecr.us-east-2.amazonaws.com/myapp` with your ECR repository

`$ aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-2.amazonaws.com/myapp`

6. Go to Github-project page - Settings - Secrets, add below key value pairs so that Github Actions would be able to fetch these configurations
   - AWS_ACCESS_KEY_ID: in the IAM credentials
   - AWS_SECRET_ACCESS_KEY: in the IAM credentials
   - HOST: EC2 public ipv4 dns
   - PORT: SSH port, 22 by default
   - SSH KEY: The generated ssh key pair: Please copy the content of xxx.pem
   - USERNAME: ec2-user by default for Amazon Linux OS, other systems please refer to https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html#connection-prereqs-private-key
