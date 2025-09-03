from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


def main() -> None:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.CODEXCLI,
        model_type=ModelType.STUB,
    )

    # Single message demo; the content will be a placeholder that includes
    # the isolated session info (work dir and session id).
    resp = model.run([{"role": "user", "content": "hello"}])
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

