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

create_namespace:
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	kubectl create namespace $(NAMESPACE)

create_volume_claim:
	@echo "$${PVC_YAML}" | envsubst | kubectl -n $(NAMESPACE) apply -f -

helm_repo:
	helm repo add tmp-rascil-helm https://gitlab.com/ska-telescope/sdp/ska-sdp-helmdeploy-charts/-/raw/master/chart-repo
	helm repo update

install_chart:
	# test the latest version of the images, published in GitLab registry
	helm install $(RELEASE) tmp-rascil-helm/dask -n $(NAMESPACE) \
	-f $(GIT_ROOT)/docker/kubernetes/values.yaml \
	--set image=registry.gitlab.com/ska-telescope/external/rascil/rascil-full:latest \
	--set imagePullPolicy=Always --set jupyter.image.tag=latest --set jupyter.image.pullPolicy=Always \
	--set jupyter.image.repository=registry.gitlab.com/ska-telescope/external/rascil/rascil-notebook \
	--wait --timeout 600s

test_k8s:
	pytest $(GIT_ROOT)/docker/kubernetes/test_k8s.py

test_k8s_jupyter:
	export JUPYTER_POD
	export SCHEDULER_IP
	export WORKER_IPS
	# we copy the test file into /mnt/data, which is a mounted volume in the pod
	kubectl cp $(GIT_ROOT)/docker/kubernetes/test_k8s_dask.py $(NAMESPACE)/$(JUPYTER_POD):/mnt/data
	# pytest is not installed inside the container by default
	kubectl -n $(NAMESPACE) exec $(JUPYTER_POD) -- bash -c \
	'pip install pytest; WORKER_IPS="${WORKER_IPS}" pytest /mnt/data/'
