
import {
  GoBackTool,
  GoForwardTool,
  NavigateTool,
  PressKeyTool,
  WaitTool,
  navigateSchema,
  goBackSchema,
  goForwardSchema,
  pressKeySchema,
  waitSchema
} from "./schemas";

import { captureAriaSnapshot } from "@/utils/aria-snapshot";

import type { Tool, ToolFactory } from "./tool";

export const navigate: ToolFactory = (snapshot) => ({
  schema: {
    name: NavigateTool.shape.name.value,
    description: NavigateTool.shape.description.value,
    inputSchema: navigateSchema,
  },
  handle: async (context, params) => {
    const { url } = NavigateTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_navigate", { url });
    if (snapshot) {
      return captureAriaSnapshot(context);
    }
    return {
      content: [
        {
          type: "text",
          text: `Navigated to ${url}`,
        },
      ],
    };
  },
});

export const goBack: ToolFactory = (snapshot) => ({
  schema: {
    name: GoBackTool.shape.name.value,
    description: GoBackTool.shape.description.value,
    inputSchema: goBackSchema,
  },
  handle: async (context) => {
    await context.sendSocketMessage("browser_go_back", {});
    if (snapshot) {
      return captureAriaSnapshot(context);
    }
    return {
      content: [
        {
          type: "text",
          text: "Navigated back",
        },
      ],
    };
  },
});

export const goForward: ToolFactory = (snapshot) => ({
  schema: {
    name: GoForwardTool.shape.name.value,
    description: GoForwardTool.shape.description.value,
    inputSchema: goForwardSchema,
  },
  handle: async (context) => {
    await context.sendSocketMessage("browser_go_forward", {});
    if (snapshot) {
      return captureAriaSnapshot(context);
    }
    return {
      content: [
        {
          type: "text",
          text: "Navigated forward",
        },
      ],
    };
  },
});

export const wait: Tool = {
  schema: {
    name: WaitTool.shape.name.value,
    description: WaitTool.shape.description.value,
    inputSchema: waitSchema,
  },
  handle: async (context, params) => {
    const { time } = WaitTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_wait", { time });
    return {
      content: [
        {
          type: "text",
          text: `Waited for ${time} seconds`,
        },
      ],
    };
  },
};

export const pressKey: Tool = {
  schema: {
    name: PressKeyTool.shape.name.value,
    description: PressKeyTool.shape.description.value,
    inputSchema: pressKeySchema,
  },
  handle: async (context, params) => {
    const { key } = PressKeyTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_press_key", { key });
    return {
      content: [
        {
          type: "text",
          text: `Pressed key ${key}`,
        },
      ],
    };
  },
};
