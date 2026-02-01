"""GPU node pipeline - executes nodes in order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artifice.engine.context import GPUContext
    from artifice.engine.node import GPUNode
    from artifice.engine.texture import Texture


class Pipeline:
    """Manages and executes a chain of GPU nodes.

    Simple linear pipeline for now - nodes execute in order,
    each node's output feeds the next node's input.
    """

    def __init__(self, ctx: GPUContext):
        """Initialize pipeline.

        Args:
            ctx: GPU context for shader execution
        """
        self._ctx = ctx
        self._nodes: list[GPUNode] = []
        self._width = 512
        self._height = 512
        self._compiled = False

    def add_node(self, node: GPUNode) -> None:
        """Add a node to the pipeline.

        Args:
            node: GPU node to add
        """
        self._nodes.append(node)
        self._compiled = False

    def remove_node(self, node: GPUNode) -> None:
        """Remove a node from the pipeline.

        Args:
            node: GPU node to remove
        """
        if node in self._nodes:
            node.release()
            self._nodes.remove(node)
            self._compiled = False

    def clear(self) -> None:
        """Remove all nodes."""
        for node in self._nodes:
            node.release()
        self._nodes.clear()
        self._compiled = False

    def set_output_size(self, width: int, height: int) -> None:
        """Set the output resolution.

        Args:
            width: Output width
            height: Output height
        """
        if width != self._width or height != self._height:
            self._width = width
            self._height = height
            self._compiled = False

    @property
    def nodes(self) -> list[GPUNode]:
        """Get list of nodes."""
        return self._nodes

    @property
    def size(self) -> tuple[int, int]:
        """Get output size."""
        return (self._width, self._height)

    def _compile(self) -> None:
        """Compile all nodes and allocate textures."""
        for node in self._nodes:
            node.compile(self._ctx)
            node.allocate(self._ctx, self._width, self._height)
        self._compiled = True

    def execute(self) -> Texture | None:
        """Execute the pipeline and return the final output.

        Returns:
            Output texture from the last node, or None if pipeline is empty
        """
        if not self._nodes:
            return None

        if not self._compiled:
            self._compile()

        # Execute nodes in order, chaining outputs to inputs
        prev_output: Texture | None = None

        for node in self._nodes:
            # Connect previous output to this node's input
            if prev_output and node.input_slots():
                node.set_input(node.input_slots()[0], prev_output)

            # Execute node
            node.execute(self._ctx)

            # Get output for next node
            if node.output_slots():
                prev_output = node.get_output(node.output_slots()[0])

        # Memory barrier after all nodes
        self._ctx.memory_barrier()

        return prev_output

    def release(self) -> None:
        """Release all GPU resources."""
        self.clear()
