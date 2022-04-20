NAMESPACE ?= test-rascil
RELEASE ?= test
GIT_ROOT = `git rev-parse --show-toplevel`

define JUPYTER_POD
$(shell kubectl -n $(NAMESPACE) get pods -o name | sed 's/pod\///g' | grep jupyter)
endef

define SCHEDULER_POD
$(shell kubectl -n $(NAMESPACE) get pods -o name | sed 's/pod\///g' | grep scheduler)
endef

define WORKER_PODS
$(shell kubectl -n $(NAMESPACE) get pods -o name | sed 's/pod\///g' | grep worker)
endef

define SCHEDULER_IP
$(shell kubectl -n $(NAMESPACE) get pods $(SCHEDULER_POD) -o json | jq '.items[].status.podIP')
endef

define WORKER_IPS
$(shell kubectl -n $(NAMESPACE) get pods $(WORKER_PODS) -o json | jq '.items[].status.podIP')
endef

define PVC_YAML
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-rascil-cip
spec:
  storageClassName: nfss1
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
endef

export PVC_YAML

namespace:
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	kubectl create namespace $(NAMESPACE)

pvc:
	@echo "$${PVC_YAML}" | envsubst | kubectl apply -f -

helm_repo:
	helm repo add tmp-rascil-helm https://gitlab.com/ska-telescope/sdp/ska-sdp-helmdeploy-charts/-/raw/master/chart-repo
	helm repo update

install_chart:
	helm install $(RELEASE) tmp-rascil-helm/dask -n $(NAMESPACE) -f $(GIT_ROOT)/docker/kubernetes/values.yaml --wait --timeout 600s

clean_up_k8s:
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	helm repo remove tmp-rascil-helm

test_k8s:
	pytest -m outside $(GIT_ROOT)/docker/kubernetes/test_k8s.py

test_k8s_jupyter:
	export JUPYTER_POD
	export SCHEDULER_IP
	export WORKER_IPS
	# we copy the test file into /mnt/data, which is a mounted volume in the pod
	kubectl cp $(GIT_ROOT)/docker/kubernetes/test_k8s.py $(NAMESPACE)/$(JUPYTER_POD):/mnt/data
	# pytest and kubernetes are not installed inside the container by default
	kubectl -n $(NAMESPACE) exec $(JUPYTER_POD) -- bash -c \
	'pip install pytest kubernetes; WORKER_IPS="${WORKER_IPS}" pytest -m inside /mnt/data/'
