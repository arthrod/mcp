
import {
  ClickTool,
  DragTool,
  HoverTool,
  SelectOptionTool,
  SnapshotTool,
  TypeTool,
  clickSchema,
  dragSchema,
  hoverSchema,
  selectOptionSchema,
  snapshotSchema,
  typeSchema
} from "./schemas";

import type { Context } from "@/context";
import { captureAriaSnapshot } from "@/utils/aria-snapshot";

import type { Tool } from "./tool";

export const snapshot: Tool = {
  schema: {
    name: SnapshotTool.shape.name.value,
    description: SnapshotTool.shape.description.value,
    inputSchema: snapshotSchema,
  },
  handle: async (context: Context) => {
    return await captureAriaSnapshot(context);
  },
};

export const click: Tool = {
  schema: {
    name: ClickTool.shape.name.value,
    description: ClickTool.shape.description.value,
    inputSchema: clickSchema,
  },
  handle: async (context: Context, params) => {
    const validatedParams = ClickTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_click", validatedParams);
    const snapshot = await captureAriaSnapshot(context);
    return {
      content: [
        {
          type: "text",
          text: `Clicked "${validatedParams.element}"`,
        },
        ...snapshot.content,
      ],
    };
  },
};

export const drag: Tool = {
  schema: {
    name: DragTool.shape.name.value,
    description: DragTool.shape.description.value,
    inputSchema: dragSchema,
  },
  handle: async (context: Context, params) => {
    const validatedParams = DragTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_drag", validatedParams);
    const snapshot = await captureAriaSnapshot(context);
    return {
      content: [
        {
          type: "text",
          text: `Dragged "${validatedParams.startElement}" to "${validatedParams.endElement}"`,
        },
        ...snapshot.content,
      ],
    };
  },
};

export const hover: Tool = {
  schema: {
    name: HoverTool.shape.name.value,
    description: HoverTool.shape.description.value,
    inputSchema: hoverSchema,
  },
  handle: async (context: Context, params) => {
    const validatedParams = HoverTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_hover", validatedParams);
    const snapshot = await captureAriaSnapshot(context);
    return {
      content: [
        {
          type: "text",
          text: `Hovered over "${validatedParams.element}"`,
        },
        ...snapshot.content,
      ],
    };
  },
};

export const type: Tool = {
  schema: {
    name: TypeTool.shape.name.value,
    description: TypeTool.shape.description.value,
    inputSchema: typeSchema,
  },
  handle: async (context: Context, params) => {
    const validatedParams = TypeTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_type", validatedParams);
    const snapshot = await captureAriaSnapshot(context);
    return {
      content: [
        {
          type: "text",
          text: `Typed "${validatedParams.text}" into "${validatedParams.element}"`,
        },
        ...snapshot.content,
      ],
    };
  },
};

export const selectOption: Tool = {
  schema: {
    name: SelectOptionTool.shape.name.value,
    description: SelectOptionTool.shape.description.value,
    inputSchema: selectOptionSchema,
  },
  handle: async (context: Context, params) => {
    const validatedParams = SelectOptionTool.shape.arguments.parse(params);
    await context.sendSocketMessage("browser_select_option", validatedParams);
    const snapshot = await captureAriaSnapshot(context);
    return {
      content: [
        {
          type: "text",
          text: `Selected option in "${validatedParams.element}"`,
        },
        ...snapshot.content,
      ],
    };
  },
};
