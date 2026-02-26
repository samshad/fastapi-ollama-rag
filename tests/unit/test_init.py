from fastapi_ollama_rag import main


def test_main_function_runs(capsys):
    """Verify the main() entry point exists and runs without error."""
    main()
    captured = capsys.readouterr()
    assert "Hello from fastapi-ollama-rag!" in captured.out


