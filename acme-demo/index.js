#!/usr/bin/env node

import { FastMCP } from "fastmcp";
import { z } from "zod"; // Or any validation library that supports Standard Schema

const server = new FastMCP({
  name: "Flight Info Bot",
  version: "1.0.0",
});

server.addTool({
  name: "FlightInfoBot",
  description: "Returns flight information based on city.",
  parameters: z.object({
    a: z.string(),
  }),
  execute: async (args) => {
    if (args.a == "Seattle") {
        return "DL2478 Departing at 10:00 AM";
      } else if (args.a == "New York") {
        return "DL1001 Departing at 12:45 PM";
      } else {
        return "Please provide a valid city.";
      }
    }
});

server.listRoots = async () => {
  return []; // empty list means: no roots supported
};

server.start({
  transportType: "stdio",
});
