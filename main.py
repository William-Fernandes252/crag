from crag import graph


def main():
    graph.get_graph().draw_png("graph.png")
    print(graph.invoke({"question": "What is agent memory?"}))


if __name__ == "__main__":
    main()
