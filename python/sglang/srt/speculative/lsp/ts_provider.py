from tree_sitter import Language, Parser, Node, Tree


def print_node(node: Node, depth: int):
    print("   " * depth + f"[{node.type}] {node.text}")


class TreeSitterCompletionProvider:
    def __init__(self, lang: str):
        if lang == "python":
            import tree_sitter_python as tspython

            self.lang = Language(tspython.language())
        elif lang == "rust":
            import tree_sitter_rust as tsrust

            self.lang = Language(tsrust.language())
        elif lang == "java":
            import tree_sitter_java as tsjava

            self.lang = Language(tsjava.language())
        else:
            raise ValueError(f"Unsupported language: {lang}")
        self.parser = Parser(self.lang)
        self.code = ""
        self.tree = self.parser.parse(b"")

    def reset_code(self, code: str):
        self.code = code
        self.tree = self.parser.parse(self.code.encode())

    def append_code(self, append: str):
        self.code += append
        self.tree = self.parser.parse(self.code.encode(), self.tree)

    def complete(self):
        assert self.tree
        cursor = self.tree.walk()

        last_child = None
        while cursor.goto_last_child():
            last_child = cursor.node

        if last_child is None:
            return []

        state = self.lang.next_state(last_child.parse_state, last_child.grammar_id)
        iter = self.lang.lookahead_iterator(state)
        symbols = iter.symbols()
        candidates = [
            self.lang.node_kind_for_id(s)
            for s in symbols
            if self.lang.node_kind_is_visible(s) and not self.lang.node_kind_is_named(s)
        ]
        if not self.code.endswith((" ", "\n")):
            candidates = [c if not c[0].isalnum() else " " + c for c in candidates]
        candidates = candidates[:5]
        return candidates


if __name__ == "__main__":
    source_code = b"""def a1_very_long_function_name():
    print("Hello, World!")

def a"""

    tsc = TreeSitterCompletionProvider("python")
    tsc.reset_code(source_code.decode())
    print(tsc.complete())

    tsc.append_code("(x")
    print(tsc.complete())

    source_code = b"""public static void main(String[] args) {
        String url = "https://jsonplaceholder.typicode.com/posts/1";
        String response = sendGetRequest("""
    tsc = TreeSitterCompletionProvider("java")
    tsc.reset_code(source_code.decode())
    print(tsc.complete())

    tsc.append_code("(x")
    print(tsc.complete())

    source_code = b"""fn main() {
    println!()"""
    tsc = TreeSitterCompletionProvider("rust")
    tsc.reset_code(source_code.decode())
    print(tsc.complete())

    tsc.append_code("(x")
    print(tsc.complete())
