.. _rascil_workflows_serial:

.. py:currentmodule:: rascil.workflows.serial


serial
======

Serial workflows are executed immediately, and should produce the same results as rsexecute workflows to within numerical
precision. Only a limited number of rsexecute workflows have been reproduced as serial workflows. The motivation is
that the scaling behaviour can be different.

For example::

        from rascil.workflows import invert_list_serial_workflow, deconvolve_list_serial_workflow
        dirty_imagelist = invert_list_serial_workflow(vis_list, model_imagelist, context='ng',
                                                      dopsf=False, normalize=True)
        psf_imagelist = invert_list_serial_workflow(vis_list, self.model_imagelist, context='ng',
                                                    dopsf=True, normalize=True)
        dec_imagelist = deconvolve_list_serial_workflow(dirty_imagelist, psf_imagelist, model_imagelist, niter=1000,
                                                           fractional_threshold=0.01, scales=[0, 3],
                                                           algorithm='mmclean', nmoment=3, nchan=self.freqwin,
                                                           threshold=0.1, gain=0.7)


.. toctree::
   :maxdepth: 1

.. automodapi::    rascil.workflows.serial.imaging
   :no-inheritance-diagram:


