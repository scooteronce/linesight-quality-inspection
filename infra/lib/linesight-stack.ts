import * as path from "node:path";
import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as agentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";

export class LineSightStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const visionModelId = "anthropic.claude-opus-5-5";
    const bedrockClient = "mantle";
    const embeddingModelId = "amazon.titan-embed-text-v2:0";

    const artifact = agentcore.AgentRuntimeArtifact.fromCodeAsset({
      path: path.resolve(__dirname, "../../runtime/bundle"),
      runtime: agentcore.AgentCoreRuntime.PYTHON_3_13,
      // AgentCore's managed Python runtime launches the configured script itself.
      entrypoint: ["main.py"],
    });

    const runtime = new agentcore.Runtime(this, "InspectionRuntime", {
      runtimeName: "LineSightInspector",
      description: "Inspects manufacturing product images and ranks similar defect references",
      agentRuntimeArtifact: artifact,
      networkConfiguration: agentcore.RuntimeNetworkConfiguration.usingPublicNetwork(),
      lifecycleConfiguration: {
        idleRuntimeSessionTimeout: cdk.Duration.minutes(5),
        maxLifetime: cdk.Duration.minutes(30),
      },
      environmentVariables: {
        AWS_REGION: this.region,
        VISION_MODEL_ID: visionModelId,
        BEDROCK_CLIENT: bedrockClient,
        EMBEDDING_MODEL_ID: embeddingModelId,
      },
    });

    // The Anthropic SDK's Bedrock endpoint authorizes against bedrock-mantle, not
    // the classic bedrock:InvokeModel action.
    runtime.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: "InvokeClaudeThroughMantle",
      actions: ["bedrock-mantle:CreateInference"],
      resources: [`arn:${cdk.Aws.PARTITION}:bedrock-mantle:${this.region}:${this.account}:project/default`],
    }));

    // Anthropic Claude models are scoped with a wildcard so VISION_MODEL_ID can be
    // switched (for example to anthropic.claude-haiku-5-5) without a stack change.
    // Cross-Region inference profiles route to US destination Regions, so include
    // the profile ARN and the foundation model ARNs in those Regions.
    runtime.role.addToPrincipalPolicy(new iam.PolicyStatement({
      sid: "InvokeInspectionAndEmbeddingModels",
      actions: ["bedrock:InvokeModel"],
      resources: [
        `arn:${cdk.Aws.PARTITION}:bedrock:${this.region}:${this.account}:inference-profile/us.anthropic.claude-*`,
        `arn:${cdk.Aws.PARTITION}:bedrock:us-east-1::foundation-model/anthropic.claude-*`,
        `arn:${cdk.Aws.PARTITION}:bedrock:us-east-2::foundation-model/anthropic.claude-*`,
        `arn:${cdk.Aws.PARTITION}:bedrock:us-west-2::foundation-model/anthropic.claude-*`,
        `arn:${cdk.Aws.PARTITION}:bedrock:${this.region}::foundation-model/${embeddingModelId}`,
      ],
    }));

    new cdk.CfnOutput(this, "AgentRuntimeArn", { value: runtime.agentRuntimeArn });
    new cdk.CfnOutput(this, "VisionModelId", { value: visionModelId });
  }
}
