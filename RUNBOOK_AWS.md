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

Every launch declares both a spend `--category` and an underlying
`--workload`. They are identical for normal runs. A bounded corrective retry
uses `--category corrective_run` but must retain the actual workload, such as
`--workload shallow_training`; the launcher applies both hardware matrices, so
the corrective budget cannot turn a two-GPU Shallow job into a one-GPU job.
AMI selection is workload-scoped and has no command-line override. Workload
`evaluation` deterministically selects Amazon AMI
`ami-06517bc7fad3c6a48`, exact name
`Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260403`; every
other workload selects the pinned base AMI. Preflight queries the selected ID
with owner `amazon` and requires owner ID `898082745236` (plus owner alias when
returned), exact name, `available` state, `x86_64`, `Linux/UNIX`, HVM, and root
device `/dev/sda1`. The selected ID remains in the printed plan and launch
request.

```bash
python3 scripts/repro_aws_launch.py \
  --category workbench_setup \
  --workload workbench_setup \
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
  --workload shallow_training \
  --instance-type g7e.12xlarge \
  --hours 12 \
  --label shallow-libero-pilot \
  --command-file /tmp/libero-shallow-2k-01.command.sh
```

Shallow training is intentionally launch-guarded to `g7e.12xlarge`: its two
local GPUs match the documented `torchrun --nproc-per-node=2` contract.  If the
first AZ reports insufficient capacity, rerender the same launch with
`--subnet-id subnet-0b49463b4ceee4d0f`, the other foundation-pinned subnet in
which preflight found G7e. Do not substitute a
`g7e.4xlarge`; it has one GPU, and neither the worker nor launcher implements a
two-node rendezvous.  If both pinned AZs are unavailable, record an AWS
capacity outage and retry later rather than launching a command that cannot
run.

The manual TensorRT replay is intentionally different from an ephemeral
training worker. It needs one bounded G7e session to survive a failed or
completed bootstrap while export, engine build, numerical validation, latency,
and compiled rollout commands are iterated over SSM on the same GPU. Plan it
with the explicit retention flag:

```bash
python3 scripts/repro_aws_launch.py \
  --category export_compile_quantize \
  --workload export_compile_quantize \
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
  --workload evaluation \
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
25. The base AMI `ami-01901bc01d5d9bb55` exposes NVIDIA driver `595.71.05`. An untouched Isaac Lab 2.2.0 base reproduced the NVIDIA-known Isaac Sim RTX startup crash with `enable_cameras=True`, while otherwise identical non-camera startup succeeded. This isolates the failure to the camera/RTX driver path rather than the RoboLab or policy changes. Evaluation launches are therefore workload-pinned to Amazon AMI `ami-06517bc7fad3c6a48`, whose AWS release notes specify Ubuntu 22.04, NVIDIA driver `580.126.09`, and G6e/G7e support.
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
47. Source commit `9eaede81032b8cbeb2c2a8844c2386e4798fc352` produced both policy images without launching another instance. The DROID v3 build completed under SSM command `47a95ddc-641f-41b8-840a-c0123e3629b7` in 3m02s and passed the real PyAV decode plus exact LeRobot 0.4.3 dependency gate. Before publication, LIBERO v2 and DROID v3 each passed OCI-label checks and a network-disabled L40S smoke covering Torch CUDA 12.8, JAX CUDA, PyAV, the runtime-specific LeRobot import, and an ONNX CUDA Add with CPU fallback disabled. ECR returned `ImageNotFound` for both commit-qualified tags immediately before the one permitted push. The immutable LIBERO digest is `sha256:ed15a0c3bdb75180a8cc2a92f9b5f7231c7e868b299e8b244ed4c8d8b2899228` (13,130,271,069 compressed bytes); the DROID digest is `sha256:39b408afa5b489b11196754ef135caf38e5d0e73b0d63cee50e7e4f3f844668e` (13,228,434,802 bytes). SSM command `d40405f8-dd14-44d4-bf7c-0aaa4a7fd9b4` explicitly pulled both registry digests, verified their RepoDigests and complete provenance labels, and repeated both network-disabled GPU smokes successfully. These artifacts remain evidence for commit `9eaede8`; if a later source-contract fix changes the OCI revision, publish new commit-qualified tags rather than relabeling or overwriting them.
48. A read-only state audit found that the local conservative cost ledger included the four-hour workbench extension while the versioned S3 ledger still contained only the first two reservations. The two remote entries were byte-for-byte identical to the first two local entries, so the complete three-entry local document was published with `If-Match` against ETag `7aeef0a6f60687ac3bee45c2ec3dbb5c`. The accepted AES256/SHA256 object is VersionId `yH285_YCNG4Ia78TW3q1tOFE.57p97lR`, SHA-256 `b8bb0171b1d473cd47c74379e0a7084d7e86c4ea7fa42a1e25263e0807065d8c`, and contains three reservations totaling `$34.54876`; a current-object round trip reproduced the hash and version history contained no delete marker. Both independent workbench deadline layers stop rather than terminate the instance. After all required state is durable in S3, explicitly terminate `i-038112fe75c610517`; otherwise its stopped 1 TiB gp3 volume at 500 MB/s continues to accrue roughly `$3.19/day`.
49. The pre-conversion launch audit found that the documented `g7e.4xlarge` Shallow fallback could not run the two-process DDP command and that the generic corrective budget category could bypass a category-only hardware guard. The launcher now separates required workload identity from spend category, applies both hardware matrices, scopes AMI selection and lifecycle to the workload, and rejects Shallow on a one-GPU G7e even when corrective funds are used. The supported capacity fallback is the same `g7e.12xlarge` in the alternate pinned AZ. The audit also found that eager-base latency depended on fixed inputs emitted only by export; official base latency is now deferred to the retained G7e session, where exact commands run all five stages for both tracks and both summarizers on one instance.
50. Before teacher conversion, adversarial output/publication review found overwriteable framework reports, a non-create-once converted-checkpoint sync, nondeterministic converted manifests, unversioned converted payload retrieval, a source-mutation window during multipart upload, indiscriminate composite-manifest metrics ingestion, and missing top-level worker metrics/cost. The corrected comparison exclusively creates both evidence files before promotion. The converted publisher now uses a deterministic manifest, claim-first/manifest-last create-once protocol, per-part and streamed whole-file SHA-256 checks, stable file-identity checks, conditional completion, exact partial recovery, version-specific round trips, durable receipts, and mandatory per-payload VersionIds in worker artifacts. Workers stage those versions individually, reject converted artifacts without pins, ingest only command-bound schema-v1 metrics, publish hash-covered training diagnostics, and record projected cost without inventing actual billing. Corrective retained sessions are authorized by the resolved export workload rather than the spend category. No checkpoint conversion or training ran while these contracts were changing. The final integrated checks passed 361 script tests, 18 data-loader tests, Ruff lint/format, Python compilation, JSON parsing, shell syntax, and `git diff --check`.
51. Reviewed source commit `229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee` was bundled with complete history into 976,233 bytes, SHA-256 `bb5a5efa2d914de5ac223a9bf251082f7de03fd2c973d19117e92d708fb854be`, and conditionally created at `s3://pi05-repro-752160877725-us-east-2/source/openpi-229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee.bundle`, VersionId `HY8r1VZTuShbxIAknhQgVyM6pVnWm9uk`. Version-specific download reproduced the hash, and the key has exactly one version with no delete marker. SSM command `6d0628cf-62c1-4574-9ced-a8b4b8c4c50f` verified the complete bundle in a new bare repository, cloned it to `/opt/pi05/source/openpi-229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee`, checked out the exact detached commit, and proved the checkout clean. The first local preflight used zsh's read-only `history` variable and stopped before any upload; the identical command was retained and replayed under Bash with a task-specific variable name. Replays use Bash and the committed publisher command.
52. The retained workbench built the final 229c policy images without launching capacity. LIBERO build command `08954174-2c86-42dd-b538-42669defff0c` produced local image ID `sha256:d76e6d73fca409e998304a6a8997f80fab1252fe0301d667a072f99dd6624f24`; DROID build command `abf33b96-ff56-4edd-9b4f-4bb371c0e24a` completed in 6m36s and produced `sha256:2afcc58cda27681892c7bbb9554e9603024c5b74f53358fad893ea876374803c`. Both passed exact dependency/config/PyAV build gates. SSM command `175f12b2-fbae-432f-aacb-292e743c43d7` then passed every OCI provenance label and, with networking disabled on the L40S, proved Torch CUDA 12.8, JAX CUDA, runtime-specific LeRobot imports, actual PyAV decode, and a no-CPU-fallback ONNX Runtime 1.26.0 CUDA Add for both images. ECR returned exactly two `ImageNotFound` failures and zero images immediately before publication. Command `b10b2d70-10cc-4388-bc0a-062adb15f450` performed the one permitted push to immutable commit-qualified tags. LIBERO is `752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:d76e6d73fca409e998304a6a8997f80fab1252fe0301d667a072f99dd6624f24` (13,130,306,741 compressed bytes), and DROID is `752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:2afcc58cda27681892c7bbb9554e9603024c5b74f53358fad893ea876374803c` (13,228,460,172 bytes). Command `991317a9-392a-4714-818f-2d0082fa2e8c` pulled both by digest, verified RepoDigests and labels, and repeated all network-disabled GPU smokes successfully. These two digest URIs and literal source commit `229c08e` are the conversion provenance even if later runbook-only commits move branch HEAD.
53. A read-only conversion-state probe first reused zsh's read-only `parameters` name locally, then sent `set -o pipefail` to the POSIX `/bin/sh` used by `AWS-RunShellScript`; neither attempt mutated checkpoint/evidence state. The retained instance was not replaced. The replay used task-specific local names and POSIX `set -eu` (or, for Bash-only runbook functions, an explicitly invoked `/bin/bash`). Command `7bcd276e-7f21-4d95-a393-402c7f941a6e` then proved both converted output directories and `/mnt/openpi/evidence` absent before first creation, with 473 GiB free. Future SSM commands carry Bash-only scripts as an encoded payload explicitly piped to `/bin/bash` rather than relying on the document's default shell.
54. LIBERO conversion command `261b2f84-ebaa-4dfa-b91d-e69ae7fc4e1a` restored the released JAX teacher and successfully wrote the BF16 PyTorch model before its attached host validation reached `python: command not found`; the DLAMI provides `python3` but no `python` alias. The output was preserved rather than reconverted. Command `127c3b55-e969-4fc9-ba90-6a2f322c17e6` validated those exact retained bytes with `python3`: 3 files / 7,473,093,598 bytes, converted revision `c73bb6ff5cbaa3c7bba5f03ea38c22bd95e8274308285e2f17b6ed2d73688dd0`, manifest SHA-256 `b1eb42ac73351d749587e3c3fd1667bc140610819e73408099d39b262cd08daa`. DROID command `10a751a7-6e99-49c2-8e9b-135023f0ac85` completed conversion and validation in 1m51s: 4 files / 7,473,096,232 bytes, revision `b4e9dcd2767b497b707d912b708729a9edd5c91bcbf402f542cd682b32c943b7`, manifest SHA-256 `ed611e4814897b84b0f138e76eed3b9caaa05ec108632a993652a69910d0b78f`. The top-level staging runbook now invokes `python3`; model commands inside the pinned images continue to use their image-provided `python`.
55. Golden-corpus commands `34fb92d2-0f01-4047-872c-a603f68ffcaa` and `ed105e0f-2e2e-425d-b01f-3ece6a1e5714` created the only canonical 64-sample corpora with split seed 42. LIBERO seed 7001 produced NPZ SHA-256 `6135b50c385431dd31fced9f44afb60a7c39bd938c5ab92e2737771685515d68` and sidecar SHA-256 `426ddd82bbb40fb25e196e6a46c54b5c0c80351845ee76acd966ea13d9f86cf3`; DROID seed 7002 produced `312525ac495fac6586412a295e7f780c3de9c0ab94c655f331cd01d718ce9c60` and `e4c779a68678b562aa5f7020eb7381ec9652468bb21dd94b6d8cf8c7ed3b8bb0`. Framework command `d40aa528-57c9-4189-ab28-a51b64e44967` passed LIBERO on every sample (minimum cosine 0.9997032287, mean 0.9999744359, MSE 0.0000642574); report SHA-256 is `548d911292ba6ba8d798adf494cc1f45f3a574cbf5dda5e0c18cb8d7d0449265` and velocity NPZ SHA-256 is `5d8426da770d6749e5fd67190dcdc0e03e546644d84412c7f8132f850af0ccc5`. Command `f138aa48-6f98-4934-9517-04d19d397aa7` passed DROID (minimum 0.9997763989, mean 0.9999911080, MSE 0.0000524981); report SHA-256 is `a62efa1a869bfe053bf6adca575186e3f3a5de27cbd4c8d0546437545d2f39a9` and velocity NPZ SHA-256 is `a66b9117daae76c17ee44be8c8b244ac998035a1099f11d58fa5abdcf51f07ee`. A first local command renderer expanded `${corpus}` before submitting SSM and made no AWS call; its corrected SSM invocation then stopped before validation because `$teacher_pytorch` was an unintended shell variable. The next exact invocation reached the validator and, before any S3 call, exposed that container-recorded `/mnt/openpi/...` paths were compared textually with host-resolved `/opt/pi05/...` paths even though `/mnt/openpi` is the verified compatibility symlink. The validator now accepts an absolute report path only when strict resolution names the exact validated regular file; it still rejects missing paths, different files, final input symlinks, changed hashes, and extra provenance fields. Focused tests include both the parent-alias acceptance and wrong-file rejection.
56. The path-identity correction and its tests were committed as control-plane commit `28076005d0331df565aa08b23292caa6ffa2cf90`; model, image, corpus, and conversion provenance deliberately remain `229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee`. Its complete 980,495-byte controller bundle has SHA-256 `f386ec2c3d1eb7a0126a83834dda79364bbcf825b3a9cd8bfa776934204e1518` and was create-once published at `source/openpi-28076005d0331df565aa08b23292caa6ffa2cf90.bundle`, VersionId `K8yDppJJfGb6hqj6dTe6LAn4O6cp4bs4`; command `4649e74f-8707-4b0b-8c91-46036b107df0` verified and cloned it cleanly. Command `87f17c51-5388-48e4-87c3-b966dbeeaa68` then passed both local evidence validators and both mutation-free upload plans. LIBERO evidence revision `2207208d27ffa46b9ba4087927355c1d7079d22a143f2350358e6fc332c24698` was create-once published by `d7c188b0-86b4-4fde-a669-3f34295b0041`; its manifest SHA-256 is `6ee67609207815150b4508d8f3664850705500e09ddf41d4af88442ed7daf4d4`, VersionId `XoGHWe7CHXfTAaSnaw9mHhzubnWpELQV`, with golden NPZ/sidecar/report/velocity VersionIds `.Vz6VEMpBieU0hpvrftb1pO2kR3xjrFg`, `VVKvEIs.Xd0OjVPOqCtHEae8Xo57rf1r`, `.TrX8fNhUDBR64ASNfUsf0E6OA4AjHaO`, and `JQJQF8QxnUiAknebXFGAVYeIyEyzSB5G`. DROID evidence revision `d8f44f39eaba43a1f5232bbf8dc57c5802b7278593b99062b7ccaaa08a86dda0` was published by `9bae9eb2-b631-4051-8dd2-676b6b1c96e8`; its manifest SHA-256 is `8a0d662814b08ec27218c69e6a97dab0932d9537bd4d57b6b427358239cd2b0c`, VersionId `4UHgPDXA0X5rJLfhjfFZDTpGno0sggr0`, with payload VersionIds `nmaVVHWiwdQAkfMEV2wMD1so4KRmwXpM`, `pQaw65EyYogv.9gNNU_3oYXRAfC7svPV`, `TUoGHRNRGGU0Sgy7q9spiot.qXTObga6`, and `2.x4LFGIlYhU2IqSTM93weSl_kUe5WZz`. Both publishers wrote AES256/SHA256 objects and the version-pinned manifest last.
57. Pre-publication execution found the converted-teacher uploader had the same textual velocity-path assumption after its otherwise successful 229c validation and dry-run. The upload had not started, so no partial converted prefix existed. The uploader now resolves the report path strictly to the adjacent validated velocity file and retains its hash gate; the wrong-file case remains rejected. The complete 981,875-byte reviewed controller at commit `effa87db4506b9881422778e3503df3a67aed930` has bundle SHA-256 `952cba72ae743962b41842d19d03298fc8d0de6e31b8dca1fda2b071f0abef9b`, S3 VersionId `RJmM6ECMjqw590vSJWi2q3_FCtwaghGe`; command `2a239793-bc61-4979-bc66-66fb510ebfe7` verified its complete history and clean checkout. Command `12ae9846-e514-45f3-aa15-f1a475b22d12` executed that reviewed controller from the clean 229c model checkout, revalidated both converted trees, and emitted two mutation-free plans. LIBERO publisher `5ad84e0b-533e-440c-af86-2a4b78463d5f` completed in 3m45s with manifest VersionId `2lhXK.lU9urPfUKPftPS._nqx_fFyTZa`; payload VersionIds are `lovWxnfPjGXaumqqymRFSooyROWg0QqA` (`norm_stats.json`), `xkFsFoARHCXyKpuweYp7sSckoQcN24Ef` (`config.json`), and `CSsniON0z0hrMv7LrnLO0v9qnPlxYWGw` (`model.safetensors`). DROID publisher `c4a2fe86-e33f-4119-b649-53b0b7ee8231` completed in 4m06s with manifest VersionId `xgmhHet70zLej9LpoJ9bWROwVYvFj5ow`; payload VersionIds are `fr9dSF.rlbM4swMPpwqzHNWOuWnQZIN_` (`norm_stats.json`), `OlJuBIA6Pv9y_lVWsI1aWaPXoF9uqhzN` (`droid.lock`), `zG8hGH6Hy2jwU8F.9Ld2st5t76HOUELc` (`config.json`), and `.mJzDLYOwUQvdlE9ORGR5T8OEsPqMuKW` (`model.safetensors`). Both multipart models were downloaded again by exact VersionId and reproduced their local full SHA-256 before receipt and manifest publication. The emitted `worker_artifact` objects, not hand-built prefixes, are mandatory inputs for Shallow workers.
58. The retained workbench deadline was extended in place by exactly three hours for the two baseline replays and durable evidence publication; no replacement instance was launched. Reservation `cfa03ba8-2a09-42c0-9b28-3f7f881c554f` adds `$9.01272`, bringing the conservative ledger total to `$43.56148`. The complete four-entry ledger is S3 VersionId `WwdchX.Da46XNc5.cVFjkU7.qqryrA7h`, SHA-256 `13eb67119d0261a58f52a2b1633e125b2ff8e47214095c70807f96e88c316db9`. Both the external EventBridge Scheduler deadline and guest systemd timer were read back at `2026-08-04T12:26:34Z`; the external action remains `StopInstances`, so the 1 TiB volume must still be explicitly terminated after all state is durable.
59. The first exact-229c LIBERO evaluator image, `sha256:a3e458626c1f0bae067c96f7fdbc08435b1d2e3ddebb77de9d1349566c97e5d7` (15,580,963,582 compressed bytes), passed provenance, CUDA, and real EGL rendering. Manual attempt 01 stopped before server readiness because UID 1000 had no passwd entry and Torch's compiler cache called `getpass`; the same retained workbench was reused with explicit `USER`, `LOGNAME`, and `TORCHINDUCTOR_CACHE_DIR`. Attempt 02 then exposed that Docker `--network none` does not provide hostname resolution for the container hostname; replays use a fixed `--hostname pi05-libero --add-host pi05-libero:127.0.0.1` while retaining the network-none boundary. Neither failure required rebuilding infrastructure or replacing the instance.
60. Attempt 03 completed the first ten LIBERO episodes at 90% success but the evaluator process remained alive after its final summary. Inspection of the retained container found the main thread waiting on a futex with 18 threads still present; the `WebsocketClientPolicy` connection was never closed. Source commit `e30480a6de404c74a996863c4fde89367350cf70` adds an explicit idempotent client close and calls it after normal LIBERO completion. Fifteen focused tests and Ruff passed. Its complete 982,931-byte bundle has SHA-256 `12f2d627e63d80631bd74b78ca3848d9a054009088867377f2cdf64493f4f9be`, S3 VersionId `D0n4i9oLFVZqpJ8JTQpMPW891l8_llI8`, and was verified into a clean detached checkout on the workbench.
61. The corrected evaluator was create-once published as `752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:51b352c1a7205d6bdae668f99060ebd05049042e1d89916993830acbdc63b374` (15,581,813,763 compressed bytes), with evaluator source `e30480a`, parent policy image `sha256:d76e6d73fca409e998304a6a8997f80fab1252fe0301d667a072f99dd6624f24`, eager backend, and exact LIBERO/LeRobot dependency labels. A digest-pulled, network-disabled L40S smoke passed CUDA, EGL, and the explicit client-close path. Attempt 04 then stopped before Docker because its wrapper passed three operands to `test ! -e`; the wrapper now performs three individual path checks. The empty output directory is retained as failure evidence and no instance was replaced.
62. Attempt 05 (`0ed5d724-31f8-4267-ade9-37ea75875efb`) exercised all four suites through the corrected close path and returned SSM `Success`. Its 40 one-trial-per-task episodes contain 39 successes (97.5%) and zero infrastructure errors: spatial 9/10, object 10/10, goal 10/10, and LIBERO-10 10/10. The run began `2026-08-04T08:21:08Z`, ended `2026-08-04T08:31:22Z`, recorded evaluator source `e30480a`, exact evaluator digest `sha256:51b352c...`, converted model revision `c73bb6ff...`, instance `i-038112fe75c610517`/`g6e.4xlarge`, and five content-hashed JSONL artifacts. This is a baseline smoke and promotion gate, not the later official 2,000-episode result.
63. The independent clean replay, attempt 06, ran under SSM command `eb79004d-7b3d-43a0-9d62-46cb663bc255` from `2026-08-04T08:32:26Z` through `08:42:34Z` and returned `Success`/0 without an intervening edit. It independently reproduced 39/40 successes (97.5%), the same 9/10 spatial and 10/10 results in each other suite, and zero infrastructure errors. Its manifest SHA-256 is `667962515bbb4bb9c14dda51dc0e479200ace8abf0ee81631c860806aeb28572`; timing SHA-256 is `45566bcf173ee1ccdc9511a8083aca544675b8662caefbfb8e432ec814820461`; replay log SHA-256 is `694deb0339e5f6384e50a97ac9bd680eba15bda8a07f812c57272cab1868d9e8`. The identical success counts across two fresh compiler caches establish the eager base as the paired local baseline and satisfy the gate for the bounded 300-step Shallow optimizer smoke.
64. After that gate passed, the reviewed 300-step LIBERO overfit controls were create-once published without launching capacity. `bootstrap/worker-bootstrap.sh` is VersionId `kHhL9j17nVdYgr3.Rott61WffK2TsgwD`, SHA-256 `8770108f5cbe662c8fc0bd9bf09c4c1820162ea7eb5c54e29d31880bfdc77052`; `specs/libero-shallow-overfit-20260804t080000z-a1-300.json` is VersionId `dqN3Bbz1fbV.WdEd_hSVsGcEet1_SrJX`, SHA-256 `7c8cc3fed42f56b37134d5b82196df16475f19609497f2e1997a7d00c05f8103`. Each key had exactly one version and no delete marker, and both exact-version heads reported AES256 plus the expected 229c source metadata. The execution-enabled bootstrap passed `bash -n`, its SHA-256 `d104154861e58c8ce991c79818dd7229ed411aecf008d554b6a4b3fc34373f00` matched the read-only launcher plan, and 80 focused launcher/worker tests passed before launch.
65. Reservation `7fa19409-dc47-43f6-ba65-13c832189e4f` launched one On-Demand `g7e.12xlarge`, instance `i-0eb2882dc5f80b010`, in `us-east-2a` for the 300-step gate. It reserves 4.25 billed hours / `$35.21584`; the conservative five-entry ledger now totals `$78.77732`, with S3 VersionId `ZvV1YyDYWbzzYi6EhjkPjCING9Py5nP3` and confirmed exact-version SHA-256 `6963bc63f692a72036e6708e0914312763d1c6d9f2c518798c5cb7b0ce9a8b52`. Live EC2 inspection confirmed ordinary On-Demand lifecycle, `CapacityReservationPreference=none`, two 97,887 MiB RTX PRO 6000 Blackwell GPUs, IMDSv2 required/hop-limit one, encrypted delete-on-termination root, and instance-initiated termination. The independent external schedule `pi05-deadline-7fa19409-dc47-43f6-ba65-13c832189e4f` targets `TerminateInstances` at `2026-08-04T12:44:41Z`.
66. That first 300-step attempt did not reach Python training. It validated and staged all 1,699 LIBERO files / 34,938,927,454 bytes, all 16 original-teacher files / 12,439,085,481 bytes, all three converted-teacher files / 7,473,093,598 bytes, and the exact policy image, then Docker exited 125 while creating the container. The generic worker bind-mounted the immutable source checkout read-only at `/workspace/openpi` and also requested a nested output bind at absent `/workspace/openpi/checkpoints`; runc could not create that mountpoint inside the read-only parent. The worker uploaded a complete terminal failure record before instance termination: log VersionId `Di.teBQaUTMalFEJ4OY8HHgYu2l7qD6S`, SHA-256 `4bb945116250532e9b8ea5d65d91f83075953e7b90856d238326aad3e4beb37f`; failed run-manifest VersionId `2QKXFUh_3zoW9MYA57sMplxvrRK.jn7_`, SHA-256 `21ddafd0629ff1236eab869db930b792cd48df2f7d7deaa6f58d181bab51abe5`; final-sync evidence VersionId `PdcDoB0nsvy7p15Hyd6U8n10gdRDTWvg`, SHA-256 `68dbf35a51e6cc8dda9885cd55868bebc44a9d843606c63e61a3c5099f4a3620`. No checkpoint exists and the instance terminated. Replays must remove the redundant nested source-tree bind in a reviewed controller while leaving model source/image provenance at 229c; never mutate the clean 229c checkout to manufacture the missing directory.
67. The two accepted eager LIBERO smokes were then sealed create-once by controller commit `3e8316a58dc714923de0404028811abde05711f2`. Its complete 1,009,563-byte bundle has SHA-256 `df8d8a65cb99f99f368b2ee0ecbed75a1d2ea324711464b002066dd9f62b200f`, S3 VersionId `2a2y_mFi7SXiYqzSiJGYOe4yPokgcAcj`, and SSM command `8eb5cc6f-9c71-4f50-b4fa-3bf8614ed501` verified a clean detached checkout. Publication command `8e20da47-d057-47ae-9cde-d7b7f7464bc0` wrote exactly 11 AES256/SHA256 objects for each run. Attempt 05 evidence revision is `8e5acd0b20ebb8cd35c57ab6de2b5dda26e239380049bbc3e1004f81e4cc5ed4`; its manifest is SHA-256 `5466047eb7f07aead181c922b9a8a6b2f4e0b74b4facdb45b86e94f7634fd8ec`, VersionId `4XpNWiVUIbri89EvDOlXHwCb0bE4yDGG`, and its S3 publication receipt is SHA-256 `e6c2742dcf096ebbde5ee0e5bbc421c7f28994506b71a94f5bfb9ee0fd304a50`, VersionId `xxtPS0qijQ0Ls.TzvC4QAEFnvxc6LQCG` (complete local publisher receipt SHA-256 `f6dba4628673038874e2f1e2addefb1b1dd37473b98476e70cd0316e643d9de7`). Attempt 06 evidence revision is `47354479aebaec0afeb93db3f8ff117b424bf7d58ed5ada1b9ee333fabf27d24`; its manifest is SHA-256 `b817d4e7aac07b92df05ddbb5dce706ab0c0c58a879832e3407c3b680cd5c81a`, VersionId `xu5IyCkYKR43x2orou5.16NcaoRdhsx1`, and its S3 publication receipt is SHA-256 `ad65e1e5acdf034eb028f7c2c29fbd534d171826392f49353441b998114ea42e`, VersionId `PkTegk3k3to4sL1nCV4JwbLAnEuv.rK9` (complete local publisher receipt SHA-256 `13c4328271084096ff5e0a134f8b59bef25e4a4083b67945267a1a765df65883`). Exact-VersionId downloads reproduced both manifest hashes; each prefix has exactly one current version of each of its 11 keys, no delete marker, and no incomplete multipart upload. Both manifests pin the pre-training cost ledger at VersionId `WwdchX.Da46XNc5.cVFjkU7.qqryrA7h`, SHA-256 `13eb67119d0261a58f52a2b1633e125b2ff8e47214095c70807f96e88c316db9`.
68. Controller commit `daf488de08a5e4894c6d581afdd585a99a9fa99c` fixes the failed worker without changing the 229c model/image provenance. It makes model and controller sources independently pinned, runs host orchestration from the controller checkout, removes the nested `/workspace/openpi/checkpoints` mount, pre-creates and validates empty mountpoints for the two intentional children of read-only `/mnt/openpi`, requires exactly one `--checkpoint-base-dir /mnt/openpi/runs`, and recognizes the real `torchrun` argv when sealing mandatory training metrics. A 1,020,930-byte bundle (SHA-256 `ca5e75dce2b0f8b40789887d9806fa8d2caa20e2e302c1a10f6cc04f26182528`, VersionId `U5zC2lK7IcmMNs3ZLT8vO.XxVWJ4fiRR`) and its 8,527-byte bootstrap (SHA-256 `f610578f4a8e51f8b9d693a9a62c19a24e025b4dd82420df7f2486b5d1586a61`, VersionId `obEQ.jINuBWUMu7xLwt8ZXSFYxiOWeXE`) were create-once published and round-tripped. A stronger fresh-clone `git fsck --full`, added before any spec publication or capacity launch, then proved that this workspace had been shallow at `15a9616`: the bundle omitted parent `c23745b5ad24e98f66967ea795a07b2588ed6c79`. The same defect exists in the legacy 229c model bundle `HY8r1VZTuShbxIAknhQgVyM6pVnWm9uk`. Both bundles can clone and check out their advertised trees, which is why `git bundle verify` alone passed, but neither is a self-contained full-history artifact. Treat `U5zC2...`, `obEQ...`, and `HY8r1...` as rejected for every future worker. They remain immutable evidence and must not be deleted, overwritten, or selected by a retry. The repository was unshallowed from the official upstream, the formerly missing parent now resolves, and local `git fsck --full --no-dangling` passes. Replacement bundles require a fresh non-shallow clone plus full fsck before publication. A separate local post-upload audit also found this workstation's `sha256sum` rejects GNU `--check --status`; no object was uploaded twice, and workstation replay compares direct digest output instead.
69. Before another paid launch, SSM preflight `b27779bd-da00-4e16-b5d4-cde42706f49e` proved the retained workbench still had a clean exact-229c model checkout, the digest-pinned LIBERO policy image, and no stale smoke path. Runtime smoke `be8e6ef0-131e-4a9c-ad61-b68bf2ad6b47` then exercised the corrected mount order against that exact image on the L40S. With the parent input mount read-only, both pre-existing child targets mounted and wrote successfully (`/mnt/openpi/runs` SHA-256 `1b368294ceab6cf10cfa0145e14d27dbf4bb9a7197a3864fb12a0e53c91a7391`; `/mnt/openpi/evidence` SHA-256 `3d8d6ab61a763b5438cb20afba7de19de2fe9cdbab9ea464b4fbf9ad9037f8dc`), an undeclared write below `/mnt/openpi` failed as required, `/workspace/openpi` remained non-writable, and Torch saw exactly one NVIDIA L40S. The exact Docker topology exited zero in 2.287 seconds. This is a mount/runtime control smoke, not a training result.

Record any future console action or command here before incorporating it into CloudFormation.
