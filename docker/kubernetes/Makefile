NAMESPACE ?= test-rascil
RELEASE ?= test
GIT_ROOT = `git rev-parse --show-toplevel`

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

install_chart:
	helm repo add tmp-rascil-helm https://gitlab.com/ska-telescope/sdp/ska-sdp-helmdeploy-charts/-/raw/master/chart-repo
	helm repo update
	helm install $(RELEASE) tmp-rascil-helm/dask -n $(NAMESPACE) -f $(GIT_ROOT)/docker/kubernetes/values.yaml --wait --timeout 600s

test_k8s_jupyter:  # TODO: doesn't work - nothing in env vars :(
	export JUP_POD=`kubectl get pods -o name | grep jupyter`
	@echo $(JUP_POD)
	export SCHEDULER_POD=`kubectl get pods -o name | grep scheduler`
	export SCHEDULER_IP=`kubectl get $(SCHEDULER_POD) --template={{.status.podIP}}`
	kubectl exec $(JUP_POD) -- bash -c \
	'python -m rascil.apps.rascil_imager --dask_scheduler $(SCHEDULER_IP):8786 --use_dask True --ingest_msname  ./data/vis/ASKAP_example.ms'

clean_up_k8s:
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	helm repo remove tmp-rascil-helm