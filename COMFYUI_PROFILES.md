# ComfyUI workflow profiles

`renderer_profiles.json` connects application-level panel data to a specific
ComfyUI API workflow. It deliberately does **not** infer node IDs or assume a
particular FLUX.2/Krea custom-node graph.

## Required inputs for the first working profile

For each workflow (Klein Base 9B and Krea 2), provide:

1. The workflow saved using ComfyUI's **Save (API Format)** command, placed at
   the `workflow_path` recorded in the profile.
2. The real node ID and input name for `positive_prompt`. This is required.
3. The real `SaveImage` output node ID in `output_node_id`.

Optional bindings may expose the common comic controls only when the exported
workflow has matching inputs:

```json
"input_map": {
  "positive_prompt": { "node_id": "6", "input_name": "text" },
  "negative_prompt": { "node_id": "7", "input_name": "text" },
  "width": { "node_id": "3", "input_name": "width" },
  "height": { "node_id": "3", "input_name": "height" },
  "steps": { "node_id": "12", "input_name": "steps" },
  "cfg": { "node_id": "12", "input_name": "cfg" },
  "seed": { "node_id": "12", "input_name": "seed" },
  "filename_prefix": { "node_id": "20", "input_name": "filename_prefix" }
}
```

Those IDs and names are an illustrative shape only; replace every value with
the values from the actual exported workflow. Do not copy them as node mapping
instructions.

The renderer clones the API workflow, applies only declared bindings, submits
it to `POST /prompt`, polls `GET /history/{prompt_id}`, downloads the first
image returned by the declared output node via `GET /view`, and writes it as
`outputs/jobs/<job-id>/panels/<panel-id>.png`.
