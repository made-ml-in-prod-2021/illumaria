# ML project for ML in Production course

## Prerequisites

* Python >= 3.7
* pip >= 19.0.3
* docker >= 4.4.4
* [minikube](https://minikube.sigs.k8s.io/docs/start/) >= 1.21.0
* [kubectl](https://kubernetes.io/docs/tasks/tools/) >= 1.21.0

## Usage

```bash
git clone https://github.com/made-ml-in-prod-2021/illumaria.git
cd illumaria
git checkout homework4
cd online_inference/kubernetes_manifests

minikube start
kubectl cluster-info
```

Apply manifests:

```bash
kubectl apply -f <manifest_file>
kubectl get pods
```
