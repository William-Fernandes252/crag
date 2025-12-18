"""The final generation stage.

It takes the "refined" knowledge (a combination of retrieved and web documents) and the user's question to produce a CYAN-colored CLI response."""

from crag.graph.nodes.chains.generator import generation_chain
from crag.graph.state import GraphState


def generate(state: GraphState) -> GraphState:
    """Generate a response based on the question and documents in the graph state.

    Args:
        state (GraphState): The current state of the graph containing the question and documents.

    Returns:
        GraphState: Updated graph state with the generated response.
    """
    question = state["question"]
    documents = state["documents"]

    # Create the prompt input
    prompt_input = {
        "question": question,
        "context": documents,
    }

    # Generate the response using the generation chain
    generation = generation_chain.invoke(prompt_input)

    return {
        "generation": generation,
        "question": question,
        **state,
    }
