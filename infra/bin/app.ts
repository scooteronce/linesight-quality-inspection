#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { LineSightStack } from "../lib/linesight-stack";

const app = new cdk.App();

// Optional custom bootstrap qualifier for accounts with a managed CDK bootstrap:
//   npx cdk deploy -c qualifier=lsightq01
// Without it, the account's default CDK bootstrap is used.
const qualifier = app.node.tryGetContext("qualifier") ?? process.env.CDK_QUALIFIER;

new LineSightStack(app, "LineSightQualityInspection", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: "us-west-2",
  },
  ...(qualifier ? { synthesizer: new cdk.DefaultStackSynthesizer({ qualifier }) } : {}),
  description: "LineSight manufacturing visual quality inspection demo",
});
