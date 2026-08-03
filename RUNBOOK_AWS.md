# Manual AWS foundation runbook

This runbook records the foundation created for the AWS-only π0.5 reproduction. It is intentionally manual: no CloudFormation should replace it until two clean smoke replays finish without undocumented changes.

## Scope and invariants

- Account: `752160877725`
- Region: `us-east-2`
- Project tag: `Project=pi05-aws-repro`
- Capacity policy: On-Demand only
- Created by this foundation pass: S3, ECR, IAM/SSM identity, a no-ingress security group, CloudWatch Logs, SNS/SQS budget transport, and one AWS Budget
- Not created: EC2 instances, launch templates, Spot requests, Capacity Reservations, Capacity Blocks, datasets, or container images
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
- No EC2 instance or paid capacity was created.

## Manual iteration log

1. Immediately after S3 creation, parallel `PutBucketEncryption` and `PutBucketTagging` calls returned `OperationAborted` because other bucket control-plane changes were still in flight. Both were repeated serially and verified. Replays use serial S3 control changes.
2. The first inline instance policy allowed ECR pull only. Review showed that the workbench must publish the pinned image, so repository-scoped `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, and `PutImage` were added and the inline policy was reapplied.
3. The initial SNS topic had zero consumers. An SSE-SQS queue and scoped topic subscription were added so email-free alarms are durable and inspectable.
4. The `Project` cost-allocation tag existed but was inactive. It was activated successfully for later reconciliation; AWS may take time to populate tag-filtered cost data.

Record any future console action or command here before incorporating it into CloudFormation.
