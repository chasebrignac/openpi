# Manual AWS foundation runbook

This runbook records the foundation created for the AWS-only π0.5 reproduction. It is intentionally manual: no CloudFormation should replace it until two clean smoke replays finish without undocumented changes.

## Scope and invariants

- Account: `752160877725`
- Region: `us-east-2`
- Project tag: `Project=pi05-aws-repro`
- Capacity policy: On-Demand only
- Created by this foundation pass: S3, ECR, IAM/SSM identity, a no-ingress security group, CloudWatch Logs, SNS/SQS budget transport, and one AWS Budget
- Live manual resource: one On-Demand `g6e.4xlarge` workbench launched for the first replay; its exact instance, cost reservation, and independent stop schedule are recorded below.
- Not created: launch templates, Spot requests, Capacity Reservations, Capacity Blocks, datasets, or container images
- Projected-cost launch checks and the cross-month project ledger are the hard stop. AWS Budgets is a delayed, account-wide, monthly backstop.

The machine-readable record is `repro/aws-foundation.json`. Policy inputs are under `infra/manual/`.

## 1. Resolve identity and network before mutation

Run these checks and stop if the account or region differs:

```bash
aws sts get-caller-identity --output json
aws ec2 describe-vpcs \
  --region us-east-2 \
  --filters Name=is-default,Values=true \
  --query 'Vpcs[].{VpcId:VpcId,CidrBlock:CidrBlock,State:State}' \
  --output json
aws ec2 describe-subnets \
  --region us-east-2 \
  --filters Name=vpc-id,Values=vpc-0b8deb3b4af5473ff \
  --query 'Subnets[].{SubnetId:SubnetId,AZ:AvailabilityZone,Available:AvailableIpAddressCount,MapPublic:MapPublicIpOnLaunch}' \
  --output json
```

Resolved on 2026-08-03: default VPC `vpc-0b8deb3b4af5473ff`; subnets `subnet-02b3d2aff3f50bf4a` (`2a`), `subnet-0b49463b4ceee4d0f` (`2b`), and `subnet-06510c52d562c79fd` (`2c`).

Before every replay, inspect exact names. A nonempty result means skip the corresponding `create-*` command and reconcile it with the `put-*`, `set-*`, or verification commands below.

```bash
aws s3api list-buckets \
  --query "Buckets[?Name=='pi05-repro-752160877725-us-east-2']" \
  --output json
aws ecr describe-repositories \
  --region us-east-2 \
  --query "repositories[?repositoryName=='pi05-repro']" \
  --output json
aws iam list-roles \
  --query "Roles[?RoleName=='pi05-repro-ssm-instance-role']" \
  --output json
aws iam list-instance-profiles \
  --query "InstanceProfiles[?InstanceProfileName=='pi05-repro-ssm-instance-profile']" \
  --output json
aws ec2 describe-security-groups \
  --region us-east-2 \
  --filters Name=vpc-id,Values=vpc-0b8deb3b4af5473ff Name=group-name,Values=pi05-repro-workbench-no-ingress \
  --output json
aws logs describe-log-groups \
  --region us-east-2 \
  --log-group-name-prefix /pi05-repro/jobs \
  --output json
aws budgets describe-budgets \
  --account-id 752160877725 \
  --query "Budgets[?BudgetName=='pi05-repro-hard-cap']" \
  --output json
```

## 2. Create and harden versioned artifact storage

The create command was run once because the preflight result was empty:

```bash
aws s3api create-bucket \
  --bucket pi05-repro-752160877725-us-east-2 \
  --region us-east-2 \
  --create-bucket-configuration LocationConstraint=us-east-2 \
  --object-ownership BucketOwnerEnforced
```

Apply controls serially. Serial order matters immediately after bucket creation.

```bash
aws s3api put-public-access-block \
  --bucket pi05-repro-752160877725-us-east-2 \
  --region us-east-2 \
  --public-access-block-configuration file://infra/manual/s3-public-access-block.json
aws s3api put-bucket-versioning \
  --bucket pi05-repro-752160877725-us-east-2 \
  --region us-east-2 \
  --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption \
  --bucket pi05-repro-752160877725-us-east-2 \
  --region us-east-2 \
  --server-side-encryption-configuration file://infra/manual/s3-encryption.json
aws s3api put-bucket-policy \
  --bucket pi05-repro-752160877725-us-east-2 \
  --region us-east-2 \
  --policy file://infra/manual/s3-tls-policy.json
aws s3api put-bucket-lifecycle-configuration \
  --bucket pi05-repro-752160877725-us-east-2 \
  --region us-east-2 \
  --lifecycle-configuration file://infra/manual/s3-lifecycle.json
aws s3api put-bucket-tagging \
  --bucket pi05-repro-752160877725-us-east-2 \
  --region us-east-2 \
  --tagging 'TagSet=[{Key=Project,Value=pi05-aws-repro},{Key=ManagedBy,Value=manual-runbook},{Key=Environment,Value=reproduction}]'
```

The lifecycle rule only aborts incomplete multipart uploads after seven days. It does not expire datasets, checkpoints, manifests, or object versions.

## 3. Create the immutable container repository

```bash
aws ecr create-repository \
  --region us-east-2 \
  --repository-name pi05-repro \
  --image-tag-mutability IMMUTABLE \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256 \
  --tags Key=Project,Value=pi05-aws-repro Key=ManagedBy,Value=manual-runbook
aws ecr put-lifecycle-policy \
  --region us-east-2 \
  --repository-name pi05-repro \
  --lifecycle-policy-text file://infra/manual/ecr-lifecycle.json
```

Only untagged intermediate images expire. Training and evaluation commands must use an immutable image digest.

## 4. Create the SSM instance identity

```bash
aws iam create-role \
  --role-name pi05-repro-ssm-instance-role \
  --assume-role-policy-document file://infra/manual/ec2-ssm-trust.json \
  --description 'SSM-only EC2 role for pi0.5 AWS reproduction' \
  --max-session-duration 43200 \
  --tags Key=Project,Value=pi05-aws-repro Key=ManagedBy,Value=manual-runbook
aws iam attach-role-policy \
  --role-name pi05-repro-ssm-instance-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy \
  --role-name pi05-repro-ssm-instance-role \
  --policy-name pi05-repro-artifact-and-logging \
  --policy-document file://infra/manual/instance-foundation-policy.json
aws iam create-instance-profile \
  --instance-profile-name pi05-repro-ssm-instance-profile \
  --tags Key=Project,Value=pi05-aws-repro Key=ManagedBy,Value=manual-runbook
aws iam add-role-to-instance-profile \
  --instance-profile-name pi05-repro-ssm-instance-profile \
  --role-name pi05-repro-ssm-instance-role
```

The role has standard SSM agent permissions plus repository-scoped ECR push/pull, artifact-bucket read/write without object deletion, reproduction-log writes, and metrics only in the `Pi05Repro` namespace. It cannot administer infrastructure.

## 5. Create no-ingress networking and logs

```bash
aws ec2 create-security-group \
  --region us-east-2 \
  --vpc-id vpc-0b8deb3b4af5473ff \
  --group-name pi05-repro-workbench-no-ingress \
  --description 'No-ingress SSM access for pi0.5 AWS reproduction' \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Project,Value=pi05-aws-repro},{Key=ManagedBy,Value=manual-runbook}]'
aws logs create-log-group \
  --region us-east-2 \
  --log-group-name /pi05-repro/jobs \
  --tags Project=pi05-aws-repro,ManagedBy=manual-runbook
aws logs put-retention-policy \
  --region us-east-2 \
  --log-group-name /pi05-repro/jobs \
  --retention-in-days 30
```

Do not authorize any ingress rules. The security group deliberately retains default outbound IPv4 access for SSM, ECR, S3, package, and source endpoints; there are no paid VPC endpoints or NAT gateways in this foundation.

## 6. Create email-free budget alert delivery

The SNS policy admits `budgets.amazonaws.com` only from account `752160877725`. The SQS queue provides an encrypted, 14-day durable sink, so budget alarms do not require an email address.

```bash
aws sns create-topic \
  --region us-east-2 \
  --name pi05-repro-budget-alerts \
  --tags Key=Project,Value=pi05-aws-repro Key=ManagedBy,Value=manual-runbook
aws sns set-topic-attributes \
  --region us-east-2 \
  --topic-arn arn:aws:sns:us-east-2:752160877725:pi05-repro-budget-alerts \
  --attribute-name Policy \
  --attribute-value file://infra/manual/budget-topic-policy.json
aws sqs create-queue \
  --region us-east-2 \
  --queue-name pi05-repro-budget-alerts \
  --attributes MessageRetentionPeriod=1209600,SqsManagedSseEnabled=true \
  --tags Project=pi05-aws-repro,ManagedBy=manual-runbook
aws sqs set-queue-attributes \
  --region us-east-2 \
  --queue-url https://sqs.us-east-2.amazonaws.com/752160877725/pi05-repro-budget-alerts \
  --attributes file://infra/manual/budget-queue-attributes.json
aws sns subscribe \
  --region us-east-2 \
  --topic-arn arn:aws:sns:us-east-2:752160877725:pi05-repro-budget-alerts \
  --protocol sqs \
  --notification-endpoint arn:aws:sqs:us-east-2:752160877725:pi05-repro-budget-alerts \
  --attributes RawMessageDelivery=true \
  --return-subscription-arn
aws budgets create-budget \
  --account-id 752160877725 \
  --budget file://infra/manual/budget.json \
  --notifications-with-subscribers file://infra/manual/budget-notifications.json
```

The delivery-path smoke test used these exact commands. The purge removed only the received synthetic test message.

```bash
aws sns publish \
  --region us-east-2 \
  --topic-arn arn:aws:sns:us-east-2:752160877725:pi05-repro-budget-alerts \
  --subject pi05-repro-foundation-test \
  --message 'Foundation alert path test; not a spend alarm.'
aws sqs receive-message \
  --region us-east-2 \
  --queue-url https://sqs.us-east-2.amazonaws.com/752160877725/pi05-repro-budget-alerts \
  --wait-time-seconds 5 \
  --max-number-of-messages 1 \
  --attribute-names All \
  --message-attribute-names All
aws sqs purge-queue \
  --region us-east-2 \
  --queue-url https://sqs.us-east-2.amazonaws.com/752160877725/pi05-repro-budget-alerts
```

The alarms are absolute actual-spend thresholds of `$1,500`, `$2,400`, and `$2,700`, plus a forecasted-spend threshold of `$3,000`. The budget is account-wide and monthly. At validation it showed `$327.962` actual and `$496.321` forecast account spend; these are not attributed to this empty foundation. Do not subtract them from the project ledger.

Activate the project tag for later Cost Explorer reconciliation:

```bash
aws ce update-cost-allocation-tags-status \
  --region us-east-1 \
  --cost-allocation-tags-status TagKey=Project,Status=Active
```

Poll the durable alert sink during active runs:

```bash
aws sqs receive-message \
  --region us-east-2 \
  --queue-url https://sqs.us-east-2.amazonaws.com/752160877725/pi05-repro-budget-alerts \
  --wait-time-seconds 20 \
  --max-number-of-messages 10 \
  --attribute-names All \
  --message-attribute-names All
```

## 7. Verification gates

Run all checks before any instance launch:

```bash
aws s3api get-bucket-versioning --bucket pi05-repro-752160877725-us-east-2 --region us-east-2
aws s3api get-bucket-encryption --bucket pi05-repro-752160877725-us-east-2 --region us-east-2
aws s3api get-public-access-block --bucket pi05-repro-752160877725-us-east-2 --region us-east-2
aws s3api get-bucket-ownership-controls --bucket pi05-repro-752160877725-us-east-2 --region us-east-2
aws ecr describe-repositories --region us-east-2 --repository-names pi05-repro --output json
aws iam get-instance-profile --instance-profile-name pi05-repro-ssm-instance-profile --output json
aws ec2 describe-security-groups --region us-east-2 --group-ids sg-05e45a674dacd5e01 --output json
aws logs describe-log-groups --region us-east-2 --log-group-name-prefix /pi05-repro/jobs --output json
aws budgets describe-budget --account-id 752160877725 --budget-name pi05-repro-hard-cap --output json
aws budgets describe-notifications-for-budget --account-id 752160877725 --budget-name pi05-repro-hard-cap --output json
aws sns list-subscriptions-by-topic --region us-east-2 --topic-arn arn:aws:sns:us-east-2:752160877725:pi05-repro-budget-alerts --output json
aws sqs get-queue-attributes --region us-east-2 --queue-url https://sqs.us-east-2.amazonaws.com/752160877725/pi05-repro-budget-alerts --attribute-names All --output json
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::752160877725:role/pi05-repro-ssm-instance-role \
  --action-names s3:PutObject ecr:PutImage logs:PutLogEvents ec2:RunInstances \
  --resource-arns \
    arn:aws:s3:::pi05-repro-752160877725-us-east-2/test-manifest.json \
    arn:aws:ecr:us-east-2:752160877725:repository/pi05-repro \
    arn:aws:logs:us-east-2:752160877725:log-group:/pi05-repro/jobs:test \
  --output json
```

Accepted state on 2026-08-03:

- S3 versioning enabled, AES-256 default encryption, all four public-access blocks enabled, bucket-owner enforcement, and TLS-only bucket policy.
- ECR AES-256 encrypted, scan-on-push enabled, immutable tags.
- Instance profile contains exactly `pi05-repro-ssm-instance-role`; the role has the SSM managed policy and one scoped inline policy.
- Security group `sg-05e45a674dacd5e01` has zero ingress permissions.
- Log retention is 30 days.
- Four budget notifications are in `OK` state and delivered through SNS to encrypted SQS.
- SNS-to-SQS delivery was tested with the message `Foundation alert path test; not a spend alarm.` The single test message was purged after successful receipt.
- IAM simulation allowed artifact upload, ECR image publication, and job-log writes; it denied `ec2:RunInstances` and destructive artifact/repository/log actions.
- No Spot, reserved, blocked, or dedicated capacity was created. The later manual replay launched the one On-Demand workbench recorded below.

## 8. Plan and launch On-Demand instances

Use `scripts/repro_aws_launch.py` for every GPU launch. Omitting `--execute` is the default read-only mode: it verifies the STS account, pinned region, AMI, subnet, zero-ingress security group, SSM profile, and instance-type offering, then prints the exact cost and deadline plan. It does not reserve money or call `RunInstances`.

AMI selection is category-scoped and has no command-line override. Category
`evaluation` deterministically selects Amazon AMI
`ami-06517bc7fad3c6a48`, exact name
`Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260403`; every
other category selects the pinned base AMI. Preflight queries the selected ID
with owner `amazon` and requires owner ID `898082745236` (plus owner alias when
returned), exact name, `available` state, `x86_64`, `Linux/UNIX`, HVM, and root
device `/dev/sda1`. The selected ID remains in the printed plan and launch
request.

```bash
python3 scripts/repro_aws_launch.py \
  --category workbench_setup \
  --instance-type g6e.4xlarge \
  --hours 8 \
  --label pi05-workbench
```

Worker launches require a command, preferably kept in a reviewed file so shell quoting is unambiguous:

```bash
# First render `/tmp/libero-shallow-2k-01.command.sh` from the immutable
# bootstrap/spec VersionIds exactly as shown in repro/WORKER_RUNBOOK.md.
python3 scripts/repro_aws_launch.py \
  --category shallow_training \
  --instance-type g7e.12xlarge \
  --hours 12 \
  --label shallow-libero-pilot \
  --command-file /tmp/libero-shallow-2k-01.command.sh
```

The manual TensorRT replay is intentionally different from an ephemeral
training worker. It needs one bounded G7e session to survive a failed or
completed bootstrap while export, engine build, numerical validation, latency,
and compiled rollout commands are iterated over SSM on the same GPU. Plan it
with the explicit retention flag:

```bash
python3 scripts/repro_aws_launch.py \
  --category export_compile_quantize \
  --instance-type g7e.4xlarge \
  --hours 8 \
  --label pi05-trt-manual-01 \
  --scheduler-role-arn arn:aws:iam::752160877725:role/pi05-repro-scheduler-deadline-role \
  --command-file /tmp/pi05-trt-session-bootstrap.sh \
  --retain-after-command
```

`--retain-after-command` is rejected outside
`export_compile_quantize`. It only omits the job service's immediate
`ExecStopPost`; it does not broaden capacity or extend time. The instance
remains On-Demand, instance-initiated shutdown remains `terminate`, and both
the guest timer and independent EventBridge Scheduler still terminate it at
the fully reserved absolute deadline. Add `--execute` only after reviewing the
plan, command, ledger, and schedule role.

The bounded RoboLab camera smoke uses the pinned evaluation AMI and published
evaluator image through the reviewed worker command. Run this without
`--execute` first:

```bash
python3 scripts/repro_aws_launch.py \
  --category evaluation \
  --instance-type g6e.4xlarge \
  --hours 1 \
  --label robolab-r580-camera-smoke \
  --scheduler-role-arn arn:aws:iam::752160877725:role/pi05-repro-scheduler-deadline-role \
  --command-file repro/robolab-smoke-worker.sh
```

The first paid evaluation smoke completed successfully. Its immutable evidence
is recorded below; replay the same read-only plan before any later launch.

Review the plan, current `.repro/cost-ledger.json`, and the command file. Only then repeat the same invocation with the explicit mutation flag:

```bash
# Add this flag only after reviewing the read-only plan:
--execute
```

The execution path rechecks the ledger under a local file lock, reserves the projected instance cost before the paid API call, and adds 15 minutes of budget margin for boot and shutdown. It uses an idempotent EC2 client token, On-Demand capacity with `CapacityReservationPreference=none`, default tenancy, no key pair, the pinned no-ingress group and SSM profile, IMDSv2-only metadata with hop limit one, and an encrypted gp3 root volume. Ephemeral job containers run with networking disabled after their immutable inputs and image are staged, so they cannot reach IMDS or mutate AWS directly. The workbench defaults to 1 TiB and stops at its deadline; ephemeral workers default to 256 GiB and terminate. Both instance and volume receive `Project=pi05-aws-repro`, stage, and run-ID tags.

The cost guard withholds `$250` of the `$3,000` cap from EC2 reservations:
the full `$150` storage/log category plus `$100` headroom. This covers gp3,
S3, ECR, CloudWatch, and transfer charges that AWS reports only after a delay.
The live Ohio gp3 price recorded on 2026-08-03 is `$0.08/GB-month`. A stopped
workbench still accrues its 1 TiB gp3 charge, so stop is only a short pause;
terminate the workbench as soon as its durable state is verified in S3.

User data installs an absolute UTC systemd deadline before starting the supplied worker command. A definitive launch failure is reconciled by client token and cancels the reservation. If AWS cannot be queried after an ambiguous network failure, the reservation remains in `launch_unknown`; do not retry until this read-only query is conclusive:

```bash
aws ec2 describe-instances \
  --region us-east-2 \
  --filters Name=client-token,Values=RESERVATION_ID \
  --output json
```

Never manually change `launch_unknown` to `cancelled` merely because the CLI timed out. First establish that no instance, including a terminated one, has that token. Record reconciliation and any deadline/SSM issue in the manual iteration log.

### First paid manual replay

The first reviewed execution launched exactly one On-Demand workbench:

- Instance: `i-038112fe75c610517` (`g6e.4xlarge`, `us-east-2a`)
- Project reservation: `ae4bc2fd-6384-49d0-ac9a-fb5e5891ef2e`
- Reserved compute: `$18.7765` for 6.25 billed hours (six requested hours plus the launcher's margin)
- Root volume: encrypted 1 TiB gp3 `vol-0de00b022d7154c96`
- Original guest stop timer: `2026-08-04T05:26:34Z`; reviewed four-hour extension deadline: `2026-08-04T09:26:34Z`
- Independent schedule: `pi05-deadline-ae4bc2fd-6384-49d0-ac9a-fb5e5891ef2e`, updated to target `StopInstances` at the extension deadline
- Extension reservation: `6a2673de-4dd7-4ab8-91e4-90f7882c19fe`, `$12.01696` for four additional On-Demand hours on the same instance

The live preflight proved an L40S with 46,068 MiB, driver 595.71.05, Docker 29.6.2, SSM identity in account `752160877725`, bucket-encryption access, and the following storage layout. The DLAMI owns the instance-store initialization: `/dev/nvme1n1` is an LVM physical volume and `/dev/mapper/vg.01-lv_ephemeral` is already mounted read/write at `/opt/dlami/nvme`. Workers reuse that verified mount; they never format it. Every execution creates a new `pi05-runs/RUN_ID` subtree and refuses an existing run ID, so mount reuse never means output/control-state reuse.

### First paid evaluation smoke

The reviewed R580 camera-smoke execution launched one short-lived On-Demand
evaluator and terminated it after the worker exited:

- Instance: `i-011eb2c219aea0e3e` (`g6e.4xlarge`, `us-east-2a`)
- Project reservation: `96c5e984-1279-42c5-b795-ba7699683422`
- Reserved compute: `$3.7553` for 1.25 billed hours (one requested hour plus the launcher's margin)
- AMI: `ami-06517bc7fad3c6a48`; observed NVIDIA driver `580.126.09`
- Evaluator image: `752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:2d17c15e62887c9fc8b4c41b7ee3d39c4c187348eb55b4273fd24e785a3325e7`
- Independent schedule: `pi05-deadline-96c5e984-1279-42c5-b795-ba7699683422`, targeting termination at `2026-08-04T01:52:03Z`
- Versioned evidence: `s3://pi05-repro-752160877725-us-east-2/manual-smoke/robolab/20260804T005329Z-i-011eb2c219aea0e3e.log`, VersionId `AIVks2vssJ5y8yT5.WDeKbiJA0rsvngM`, SHA-256 `dba12538bda077a51f2de816196dfa54d285da47a4fbb8f6cde1d998e6794c5d`

The worker verified the exact AMI, driver, image digest, source labels, Torch
and dependency versions, then ran the four upstream RoboLab suites
`test_isaaclab`, `test_registered_envs`, `test_tasks_valid`, and
`test_run_empty`. All 128 tests passed (72 test dots through 56%, followed by
56 more through 100% in the quiet pytest output), the log records
`smoke_exit_code=0`, and EC2 entered `shutting-down` immediately after
completion and was later verified `terminated`. This is the first clean
camera-enabled evaluator smoke, not a policy-quality evaluation.

## Manual iteration log

1. Immediately after S3 creation, parallel `PutBucketEncryption` and `PutBucketTagging` calls returned `OperationAborted` because other bucket control-plane changes were still in flight. Both were repeated serially and verified. Replays use serial S3 control changes.
2. The first inline instance policy allowed ECR pull only. Review showed that the workbench must publish the pinned image, so repository-scoped `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, and `PutImage` were added and the inline policy was reapplied.
3. The initial SNS topic had zero consumers. An SSE-SQS queue and scoped topic subscription were added so email-free alarms are durable and inspectable.
4. The `Project` cost-allocation tag existed but was inactive. It was activated successfully for later reconciliation; AWS may take time to populate tag-filtered cost data.
5. The launch wrapper was exercised in read-only plan mode against the live account for a one-hour `g6e.4xlarge` workbench. It resolved the pinned AMI root device, `us-east-2a` offering, zero-ingress group, exact SSM profile, 1 TiB encrypted gp3 volume, stop deadline, and a `$3.7553` reservation including the 15-minute margin. No reservation was written and no EC2 instance was launched.
6. Pre-launch review found that worker and staging boundary checks call `GetBucketEncryption`, but the first inline role policy omitted `s3:GetEncryptionConfiguration`. The bucket-scoped read permission was added, the live inline policy was reapplied, and policy simulation plus a role-credential smoke check were required before the first paid launch.
7. The pinned training base image is hosted in AWS's `763104351884/pytorch-training` ECR repository. The first role policy scoped all repository reads to the project ECR, so exact-repository pull-only permissions were added for that AWS DLC source; project ECR remains the only repository with push permissions.
8. Container access to the instance role was unnecessary after host-side staging and syncing. The launch template's IMDS hop limit was reduced from two to one and ephemeral containers were placed on Docker's `none` network, leaving loopback available for same-container evaluation while preventing metadata or external AWS access.
9. The installed AWS CLI rejects `--expected-bucket-owner` on high-level `aws s3 cp/sync` commands. Owner/account/region/encryption are verified first with owner-aware `s3api` calls; small control objects use `s3api put-object`, while large multipart syncs omit the unsupported flag and are followed by manifest/version/hash verification.
10. The first paid workbench showed that a guest-only systemd deadline cannot bound spend if cloud-init fails. A dedicated EventBridge Scheduler role and one-time external `StopInstances`/`TerminateInstances` schedule were added; launch is considered successful only after that external deadline exists, with the guest timer retained as defense in depth.
11. The first SSM diagnostic used `set -o pipefail`, but `AWS-RunShellScript` invoked `/bin/sh`, which does not support that Bash option. Subsequent SSM diagnostics either use POSIX shell syntax or invoke Bash explicitly.
12. The next diagnostic embedded Docker's Go-template braces through multiple JSON/shell quoting layers and produced an invalid `docker info --format` expression. The replay now uses plain `docker info` unless a command is supplied through a file.
13. Live `lsblk`, `findmnt`, and `df` evidence confirmed that the pinned DLAMI preformats its 600 GB instance store as LVM/ext4 and mounts it at `/opt/dlami/nvme`. Worker discovery was changed to accept only that specifically verified layout and to refuse reformatting any mounted or otherwise initialized device.
14. The first pinned Hugging Face staging environment installed `huggingface-hub==0.33.1` successfully, but that version did not provide the newer `hf` executable. No dataset payload had transferred. The command was corrected to call `huggingface_hub.snapshot_download` directly with the exact repository revision and local directory; replays use the Python API rather than assuming a CLI entry point.
15. A raw AWS DLC GPU smoke appeared to hang because the DLC defines `bash -m dockerd_entrypoint.sh` as its entrypoint; appending `python -c ...` did not override it. The diagnostic container was cancelled and exited cleanly. Replaying with `--entrypoint python` proved Torch 2.7.1+cu128, CUDA 12.8, L40S availability, and compute capability 8.9. The reproduction Dockerfiles deliberately set `ENTRYPOINT []`, so their normal commands do not inherit this behavior.
16. Parallel unauthenticated Xet downloads used 16 DROID workers plus 8 LIBERO workers and eventually received HTTP 429 responses from Hugging Face's per-file read-token endpoint. The partial snapshots remained resumable on persistent EBS. Replays serialize the two repositories with `max_workers=1` (increasing only after observed headroom), validate the final exact commit, and never delete a partial merely because a rate limit interrupted it.
17. To overlap public GCS teacher transfer with the final local review, the two staging scripts and reproduction config were uploaded as three AES256-encrypted, versioned manual-control objects before use. `repro_stage_data.py` is VersionId `zlXnKEjbimyEOR4FgN6bXeNj.OuuSPvN`, SHA-256 `a055f508787c010e97ba9a6abe535e84aa8a00a14214e9703553e3afd0930a0f`; `repro_stage_checkpoints.py` is VersionId `iEVUbEBJcbhJ5PriMEUhBxSHAMoZ7QWJ`, SHA-256 `c473c8e561587c297e735c07bc76870ce178ba009e1abb5dbeb3ee01c98614d4`; and `reproduction.json` is VersionId `D8H8r5a.eXlfv_YirtYC9EgHk5P4zx9P`, SHA-256 `ed9b3f81de4a93660b4d5bad5ef28571db8e8bd3e58e2e1b09bd9fbb6c905395`. The workbench verified all three before running the download action. These pre-commit control versions are bootstrap evidence, not the final source artifact; the eventual committed git bundle must reproduce or supersede them before training.
18. The pinned RoboLab Git checkout reached commit `0aef241fb088ca21bb4ebd24448940ed56620d17`, but the base AMI did not include the `git-lfs` executable needed for its large simulator assets. The same checkout was retained; `git-lfs` was installed from Ubuntu packages, `git lfs install --local` and `git lfs pull` were run in place, followed by exact-HEAD and `git lfs fsck` verification. No instance replacement or fresh clone was needed.
19. Both released teachers were downloaded from their pinned public GCS prefixes, checked against complete file inventories, and uploaded to versioned S3 before conversion. LIBERO contains 16 files and 12,439,085,481 bytes; its manifest SHA-256 is `9140fa118b1a2b627726519cb3d21a0a98f2b1b736b5909a49520fc75d8dd8ad`, its source revision is the inventory hash `b00d25ec1a1284656ccfd0cf00597fced40fa20c9c7c39ebfdf256db6e844fb7`, and its S3 manifest VersionId is `oX5OL_hTQDoYmYD7bTZ.sM7.4KxB5FX3`. The DROID joint-position teacher contains 26 files and 12,435,136,033 bytes; its manifest SHA-256 is `64e4082767ac652d35828f721ca0906bd9a97f78a769a4bf4f75b09837d5bf46`, its source revision is the inventory hash `6487c08461e26cac570a2781f477474e6573c7a6e0a4ba93a9f0efb146c2db5b`, and its S3 manifest VersionId is `etUbiXvb8B6C7ltGXEfmrB96kJxI18HC`. Replays must use the version-pinned manifests and must not substitute the generic DROID teacher for `pi05_droid_jointpos`.
20. RoboLab was built from commit `0aef241fb088ca21bb4ebd24448940ed56620d17` with client commit `aa6420561529593114160d05e5ad155792b272f3` on the resolved Isaac Lab base `sha256:b4d8e96cbfb9a6c40067bec6cc5ee180e36d4c0164b25f7215c5f47e31897b94`. The first runtime check incorrectly queried `importlib.metadata` for release versions: the image's internal `isaaclab` Python distribution reports `0.44.8`, and bundled Isaac Sim has no `isaacsim` distribution metadata. The replay gate now reads `/workspace/isaaclab/VERSION` (`2.2.0`) and `/workspace/isaaclab/_isaac_sim/VERSION` (`5.0.0-rc.45+release.23960.184afb15.gl`) directly before running the GPU task tests. The image was retained and the corrected check was rerun; no instance replacement was needed.

21. A pre-launch artifact trace found that the first Shallow worker example staged only the dataset even though its config also reads the released JAX normalization assets and converted PyTorch teacher. Staging uploads now emit copy-ready `worker_artifact` objects, the example includes all three inputs, and converted teachers receive distinct content/provenance revisions and S3 prefixes.
22. Numeric training checkpoints were durably uploaded but originally had no manifest in the schema enforced for later worker inputs. Declared checkpoint/artifact outputs can now set `publish_destination`; only after their S3 receipts exist does the worker upload a worker-input manifest and record a complete copy-ready descriptor in `run-manifest.json.published_inputs`. This is the required Shallow-to-SnapFlow and model-to-evaluation handoff.
23. Container review found that the editable project install resolved normal dependencies independently of `uv.lock`, and that generated `data`, `checkpoints`, `runs`, calibration, and engine directories could enter a later Docker context. The training Dockerfile now exports from the frozen lock before installing the local packages without dependency resolution, and `.dockerignore` excludes those generated trees. Replays require a clean checkout before assigning the OCI source-revision label.
24. Early manual downloads used `/opt/pi05`, while the reviewed container and worker interface consistently uses `/mnt/openpi`. After verifying `/mnt/openpi` did not exist and `/opt/pi05` was on the persistent 1 TiB root volume, the workbench added the single compatibility symlink `/mnt/openpi -> /opt/pi05`. Container bind sources and manual commands now use `/mnt/openpi`; ephemeral workers create that path directly and do not need the symlink.
25. The base AMI `ami-01901bc01d5d9bb55` exposes NVIDIA driver `595.71.05`. An untouched Isaac Lab 2.2.0 base reproduced the NVIDIA-known Isaac Sim RTX startup crash with `enable_cameras=True`, while otherwise identical non-camera startup succeeded. This isolates the failure to the camera/RTX driver path rather than the RoboLab or policy changes. Evaluation launches are therefore category-pinned to Amazon AMI `ami-06517bc7fad3c6a48`, whose AWS release notes specify Ubuntu 22.04, NVIDIA driver `580.126.09`, and G6e/G7e support.
26. RoboLab's editable install upgraded `typing-extensions` to 4.16.0 and selected `typeguard` 4.6.0. Isaac Lab's bundled Torch 2.7.0 could no longer import `torch._dynamo`, while downgrading only `typing-extensions` made the typeguard pytest plugin fail. The same source checkout and base layers were retained; `typeguard==4.4.2` and `typing_extensions==4.12.2` were installed together without dependency resolution, and the Docker build now proves both import paths. The corrected immutable image digest is `sha256:2d17c15e62887c9fc8b4c41b7ee3d39c4c187348eb55b4273fd24e785a3325e7`.
27. After publishing that image, a host-side `ecr:DescribeImages` verification failed because the scoped instance role supported push/pull but omitted this read-only inventory action. `ecr:DescribeImages` was added only for the project repository, the live inline policy was reapplied, IAM simulation returned `allowed`, and the same workbench verified the immutable digest. No instance or image rebuild was needed.
28. The first paid R580 evaluator used reservation `96c5e984-1279-42c5-b795-ba7699683422` and instance `i-011eb2c219aea0e3e`. It observed driver `580.126.09`, pulled the exact evaluator digest, passed all 128 tests across the four camera/task smoke suites, uploaded its AES-256 encrypted versioned log with `smoke_exit_code=0`, and entered termination immediately. The external deadline remained independent protection but did not have to fire. The log VersionId and hash are in the evaluation-smoke section above.
29. The minimal Hugging Face staging virtual environment did not include the `polars` dependency needed for the DROID parquet-schema gate. The workbench retained the completed snapshot and installed the lock-pinned `polars==1.30.0` wheel into that environment. Replays use the committed reproduction image, whose frozen lock already includes this version; a separate ad hoc validation environment must install the same pin before running `repro_stage_data.py validate`.
30. The first real MolmoAct2 validation correctly stopped before hashing because the fixture-derived gate expected the two video feature names to be physical parquet columns. Inspection of the exact pinned snapshot showed the non-video policy inputs in `data/chunk-000/file-000.parquet`, while LeRobot v3 stores the declared video features under `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`. Iteration 2 (SHA-256 `34dd04cbf38d43c9c8ff89e04182e84d64733fcba8733a1bdf97fd6945bb00f6`, VersionId `jm0L2TIJ4vT6Jjdxw5EyGkm6eOvUryFu`) was superseded when review found that subtree presence alone did not prove exact episode-derived media coverage; its in-progress hash was cancelled before it emitted a manifest and the object version is tagged `Status=superseded`. An earlier byte-identical VersionId `qauY1079BttAHh1T9ETK563Jr0UB4oFa` accidentally carried placeholder SHA metadata, was never executed, and is tagged `Status=invalid`. The accepted iteration 3 derives every expected parquet and MP4 path from the pinned episode metadata, rejects missing/orphan/empty media, and pins the observed 518 exterior-left and 316 wrist-left files. Its script SHA-256 is `f09196890691a877a66091d2ba298c9a2d3f702ca84dba0f56d5e5be9fec07a3`, staged at `manual/staging/iteration-03/scripts/repro_stage_data.py`, VersionId `.o4gDDzbSV3HiEMDWHZz3f0Mt6ZIgblC`. Replays use that approved version or the superseding committed source.
31. The workbench's 1 TiB gp3 root volume was provisioned at the 125 MB/s baseline, making each full 259 GB integrity pass approximately 35 minutes from bytes divided by throughput. The same attached volume `vol-0de00b022d7154c96` was increased in place to 500 MB/s at `2026-08-04T01:29:43Z`; no data, filesystem, instance, IOPS, or volume size changed. The maximum incremental provisioned-throughput cost is about `$15` for an entire month and remains inside the `$150` non-compute reserve; the volume is terminated with the workbench after durable upload rather than left at the higher setting.
32. The serialized exact-revision LIBERO resume completed without another rate limit. Validation found 1,693 episodes, 273,465 frames, 40 tasks, 1,699 payload files, and exactly 34,938,927,454 payload bytes. The upload revalidated and rehashed the local tree, synchronized all 1,699 files under the immutable revision prefix, and produced manifest SHA-256 `a195fab74ceb43dcc50438f3401f8174a8c477f90f47385ec806048c04537f5f` at S3 VersionId `30NitaqVRt8BpxA4TAzzxG27icirmyLF`. A version-specific `GetObject` reproduced that hash and source metadata. This is the copy-ready LIBERO dataset input for worker specs.
33. Accepted DROID iteration 3 validated 74,604 episodes, 17,758,044 frames, 49,623 tasks, 50 parquet shards, 834 episode-referenced MP4 files, 1,414 total payload files, and exactly 258,712,592,431 payload bytes. It then rehashed and synchronized the same retained tree to the immutable revision prefix. The final manifest SHA-256 is `04fd47ec9acc3d211e3ed16e3f44b7f6c73c7b82ca22a98da017f37493854326`, VersionId `T0MmQScu_e_I_.n824j5XWXHFzUr6n8_`. A version-specific encrypted `GetObject` reproduced the hash and source revision; an independent S3 listing reproduced all 1,414 payload objects and the exact byte total. This is the copy-ready DROID dataset input for worker specs.
34. The existing workbench pulled the pinned `nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3` base without starting another instance. A network-disabled GPU smoke observed TensorRT `11.0.0.114`, an L40S, host driver `595.71.05`, and NVIDIA's CUDA 13.3 forward-compatibility user-mode driver `610.43.02`. The first inspection command lost Docker Go-template quoting through SSM; the retained image was inspected with `docker image inspect | jq` instead. A second smoke proved that TensorRT 11's `trtexec --version` prints its version but exits 1 with `Model missing or format not recognized`; the compiler Dockerfiles now use the TensorRT Python version assertion, `command -v trtexec`, and the successful `trtexec --help` build signature. No image or instance restart was needed.
35. The same retained image was tagged once as `tensorrt-base-26.06-7cd94ee931d2` and pushed to the immutable project ECR. Docker reported that NVIDIA digest `sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3` is a multi-platform index and published only the locally resolved amd64 child, yielding account-local digest `sha256:2a5a0a9a32ec5ddc1c384c15ddcf3b89ddc4f8647e7ee7ae708d844210183a1e` and 6,244,295,363 compressed bytes. A fresh pull by that ECR digest, with networking disabled in the container, reproduced TensorRT `11.0.0.114`, the successful `trtexec` build banner, and CUDA forward compatibility. Compiler builds use this mirror digest, never the tag or external NGC reference.
36. Source review and the retained TensorRT base inspection consumed most of the original six-hour workbench window, so a bounded four-hour extension was reserved as `6a2673de-4dd7-4ab8-91e4-90f7882c19fe` for `$12.01696`; no instance was launched. The existing EventBridge Scheduler deadline was updated and read back as enabled at `2026-08-04T09:26:34Z`, with `ActionAfterCompletion=DELETE`, ten retries, and the same `StopInstances` target. The first guest-timer edit replaced the spaces in the systemd calendar value with periods and systemd correctly refused the invalid timer; the same instance was retained, the exact `OnCalendar=2026-08-04 09:26:34 UTC` line was restored, `daemon-reload` was run, and the active timer was verified to trigger at the same deadline. EC2 still reports instance-initiated shutdown behavior `stop`.
37. The final pre-bundle adversarial review found unsafe experiment-name checkpoint deletion, stale NVMe run reuse, container-visible worker control directories, a non-create-once compiled publisher using an unsupported high-level S3 owner flag, missing clean-checkout evidence, and a bare-host DROID compiled path. The reviewed source now rejects checkpoint escapes/symlink ancestors, uses unique fresh experiment and NVMe run IDs, mounts only payload roots, creates host evidence without overwrite, conditionally publishes compiled files with exact empty/history/version gates, binds the protected clean Git identity, and wraps every DROID phase in its digest-pinned v3 runtime. These were source corrections only; no AWS object or capacity mutation occurred.
38. The payload-only mount assumption was exercised on the retained workbench with the already-local account-mirrored TensorRT image. A UID-1000 network-disabled GPU container wrote through `/output/artifacts` while `/output/.ready` was absent from its namespace; the host read the expected probe and removed the exact temporary directory with empty-directory checks. The smoke printed `payload-only-mount-smoke=passed`; no instance was launched or replaced.
39. Pre-build inspection on the same workbench confirmed the mirrored TensorRT parent is Ubuntu 24.04/Python 3.12.3/TensorRT 11.0.0.114, the training DLC is Ubuntu 22.04/Python 3.12.10/Torch 2.7.1+cu128, the LIBERO apt dependencies exist, and ONNX Runtime GPU 1.28 installs in the actual DLC with the CUDA provider exposed. The compiler publication smoke now requires a no-CPU-fallback ONNX CUDA Add plus a one-iteration `trtexec` engine/inference before push, rather than accepting imports alone.
40. Reviewed source commit `81a359322e73e6f5765dcc705c06bc9b5111e9ce` was bundled with complete history into 939,155 bytes, SHA-256 `89b719d74e8457c68785dee090262a95aea460b68c6cb5132aa8f156c5cc0d3a`, and conditionally uploaded once to `s3://pi05-repro-752160877725-us-east-2/source/openpi.bundle`, VersionId `v83TLOupHVE3l6V_qFowJguvVEkQ0WB5`. A version-specific download reproduced the hash and the key had exactly one version with no delete marker. The first two remote verification attempts used `git bundle verify` outside a Git repository (first from the SSM working directory, then against `/opt/pi05/manual-repo`, which was not a repository) and failed before cloning. The retained workbench initialized an empty bare verification repository, verified the same bundle as complete, checked its HEAD, cloned it, checked out the exact detached commit, and proved the checkout clean.
41. The first committed LIBERO v2 policy-image build reached the frozen dependency install and stopped before image creation or ECR publication: OpenPI's lock selects simulator-only `gym-aloha==0.1.1`, whose `mujoco==2.3.7` dependency has no CPython 3.12 Linux wheel, so its source build demanded an unset `MUJOCO_PATH`. The training DLC is intentionally Python 3.12, while simulation is already isolated in the separately pinned LIBERO and RoboLab images. Replays use uv's exact-lock `--prune gym-aloha` operation, which was checked to remove only that unused simulator branch (nine packages, no additions or replacements), assert that Gym-Aloha and MuJoCo are absent, import both required distillation configs, and label the simulator runtime `external`; the frozen policy/training dependency set remains unchanged otherwise.
42. Pre-execution review found that the worker bootstrap repeated the out-of-repository `git bundle verify` error and that five TensorRT runbook label checks put backslash-escaped quotes inside single-quoted Docker Go templates. The bootstrap now creates a fresh root-owned bare verification repository, verifies the bundle there, checks bundle HEAD against the signed worker spec before cloning, and refuses stale reuse. The compiler commands now pass literal valid Go templates. These corrections were committed before any worker or compiler image publication.
43. The first source object was safely create-once, but its generic `source/openpi.bundle` key could not represent a corrected commit without either overwriting the current object or accumulating ambiguous versions. The repeatable source publisher now derives `source/openpi-$SOURCE_COMMIT.bundle`, requires empty version/delete history, uses `If-None-Match: *`, verifies the exact returned VersionId and metadata, requires exactly one final version with no delete marker, and round-trips that version's SHA-256. Worker specs copy the printed commit-specific URI and VersionId together.
44. The corrected policy-image build passed the frozen dependency, tokenizer, source-install, config-import, and version gates, then exposed an AWS DLC ABI mismatch at the former TorchCodec smoke. The pinned DLC's CPython 3.12.10 is a static build (`Py_ENABLE_SHARED=0`) with only `libpython3.12.a`; lock-pinned TorchCodec 0.4.0 attempts to load `libpython3.12.so.1.0` when using the available FFmpeg 4 libraries. Both exact LeRobot revisions support an explicit PyAV backend. Training now passes `video_backend="pyav"` rather than allowing package presence to select a broken default, the image retains its otherwise exact locked graph, and build plus digest smokes generate and decode a real local video through the LeRobot PyAV path. The OCI decoder label and worker identity gate bind that choice; later pilot throughput will measure whether it is an actual input bottleneck before any broader runtime change.
45. A pre-compiler runbook audit found that the combined TensorRT policy section required post-push GPU/import smokes but specified them only in prose and resolved only the DROID digest in its executable block. Before compiler execution, the runbook was completed with digest resolution for both tracks, exact architecture/RepoDigest/provenance/toolchain/runtime label gates, and network-disabled digest smokes for Torch CUDA, no-fallback ONNX Runtime CUDA, TensorRT builder creation, tokenizer/config loading, and the policy/WebSocket protocol stack. The LIBERO combined image remains only an intermediate parent; final LIBERO export and evaluation still require the later evaluator digest.
46. The first live v2 GPU smoke passed Torch CUDA, JAX CUDA, the generated PyAV decode, and runtime imports, then correctly failed the no-CPU-fallback ONNX Runtime gate before ECR publication. PyPI's ONNX Runtime GPU 1.28 wheel requires CUDA 13 (`libcublasLt.so.13`), while the training DLC is pinned to CUDA 12.8. The official ONNX Runtime CUDA matrix states that PyPI 1.27 and newer default to CUDA 13, while 1.26.x through 1.21.x use CUDA 12.8. The same built image was retained and a temporary container replaced only ORT with 1.26.0; the complete PyAV/Torch/JAX/no-fallback ORT CUDA smoke then passed on the L40S. The policy Dockerfile now pins 1.26.0 and binds it in OCI and worker identity; the TensorRT compiler's separate CUDA 13.3/ORT 1.24.2 environment is unchanged. Reference: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

Record any future console action or command here before incorporating it into CloudFormation.
