# Data handling and security

This repository contains software only. Do not commit clinical DICOM, patient
identifiers, screenshots, annotation presentations, derived image exports,
model checkpoints, access tokens, private keys, or local environment files.

Run the tools against approved local or hospital storage and write outputs to a
separate, access-controlled directory. DICOM headers and derived outputs may
remain identifiable even when filenames are changed. Review the hospital's
privacy, retention, and transfer policy before moving any data.

The registration and reconstruction code is research software. It is not a
medical device and does not establish clinical accuracy or patient-specific
3-D truth. A passed software quality gate is not clinical approval.
