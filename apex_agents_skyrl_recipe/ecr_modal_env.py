"""Modal environment subclass for the apex recipe.

Extends OSS harbor's ``ModalEnvironment`` (import-only, no fork) with the two
capabilities the apex trial flow needs that upstream doesn't have yet:

1. ``registry_image``: a per-trial prebuilt ECR image (injected by
   ``TITOHarborGenerator`` from the task's ``archipelago.json`` ``ecr_image``).
   Routed through upstream's native prebuilt-image path: we set
   ``task_env_config.docker_image``, and upstream's ``_ModalDirect.start``
   then uses ``Image.from_aws_ecr(..., secret=Secret.from_name(registry_secret))``
   because the URL contains ``.dkr.ecr.``.

2. ``encrypted_ports`` + ``get_tunnel_url``: the Archipelago agent (also in
   this package) reaches the sandbox's MCP gateway through a Modal tunnel on
   port 8000. Ports must be declared at ``Sandbox.create`` time, and upstream's
   ``_create_sandbox`` has no extension hook for extra create-kwargs, so we
   override it with a faithful copy (+1 line). Version-coupled to the pinned
   harbor SHA (f03db62f) — re-check on harbor upgrades.

Legacy kwarg compatibility with older ``archipelago_tito.yaml`` configs:
``ecr_modal_secret_name`` -> ``registry_secret``; ``modal_app_name`` ->
``app_name``; ``ecr_aws_region`` / ``ecr_aws_profile`` are accepted and
dropped (upstream's ``from_aws_ecr`` takes the region from the image URL).
"""

from typing import Any

from harbor.environments.modal import ModalEnvironment, Sandbox


class EcrModalEnvironment(ModalEnvironment):
    def __init__(
        self,
        *args,
        registry_image: str | None = None,
        encrypted_ports: list[int] | None = None,
        # Legacy kwarg names (mapped or dropped):
        ecr_modal_secret_name: str | None = None,
        modal_app_name: str | None = None,
        ecr_aws_region: str | None = None,  # noqa: ARG002 - accepted, unused
        ecr_aws_profile: str | None = None,  # noqa: ARG002 - accepted, unused
        **kwargs,
    ):
        if ecr_modal_secret_name and "registry_secret" not in kwargs:
            kwargs["registry_secret"] = ecr_modal_secret_name
        if modal_app_name and "app_name" not in kwargs:
            kwargs["app_name"] = modal_app_name
        super().__init__(*args, **kwargs)
        self._encrypted_ports: list[int] = list(encrypted_ports or [])
        if registry_image:
            # Upstream reads task_env_config.docker_image in _ModalDirect.start;
            # should_use_prebuilt_docker_image() returns True once it's set.
            self.task_env_config.docker_image = registry_image

    async def _create_sandbox(
        self,
        *,
        entrypoint: list[str] | None = None,
        block_network: bool | None = None,
        experimental_options: dict[str, Any] | None = None,
    ) -> Sandbox:
        """Copy of ModalEnvironment._create_sandbox @ f03db62f + encrypted_ports."""
        if not self._encrypted_ports:
            return await super()._create_sandbox(
                entrypoint=entrypoint,
                block_network=block_network,
                experimental_options=experimental_options,
            )

        if block_network is None:
            block_network = self._network_disabled

        kwargs: dict[str, Any] = {}
        if experimental_options:
            kwargs["experimental_options"] = experimental_options
        if (cpu := self._cpu_config()) is not None:
            kwargs["cpu"] = cpu
        if (memory := self._memory_config()) is not None:
            kwargs["memory"] = memory
        if (gpu := self._gpu_config()) is not None:
            kwargs["gpu"] = gpu
        if self._dynamic_network:
            block_network = False
            kwargs.update(self._dynamic_network_kwargs(self.network_policy))
        elif self._network_is_allowlist:
            kwargs.update(self._allowlist_network_kwargs(self.network_policy))
        if labels := self._sandbox_labels():
            if self._sandbox_v2_enabled:
                self.logger.debug("V2 sandboxes do not support tags; dropping labels")
            else:
                kwargs["tags"] = labels
        if region := self._kwargs.get("region"):
            kwargs["region"] = region

        # --- apex addition: declare tunnel ports at create time ---
        kwargs["encrypted_ports"] = self._encrypted_ports

        if self._sandbox_v2_enabled and not hasattr(Sandbox, "_experimental_create"):
            raise RuntimeError("modal_sandbox_v2 not available, please upgrade modal")

        create_fn = Sandbox._experimental_create.aio if self._sandbox_v2_enabled else Sandbox.create.aio
        return await create_fn(
            *(entrypoint or ()),
            app=self._app,
            image=self._image,
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            name=self.session_id,
            block_network=block_network,
            secrets=self._secrets_config(),
            volumes=self._volumes_config(),  # ty: ignore[invalid-argument-type]
            **kwargs,
        )

    async def get_tunnel_url(self, port: int) -> str:
        """Public tunnel URL for a declared encrypted port (used by Archipelago)."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started. Call start() before getting tunnel URLs.")
        tunnels = await self._sandbox.tunnels.aio()
        if port not in tunnels:
            raise RuntimeError(f"No tunnel on port {port}. Available: {list(tunnels.keys())}")
        return tunnels[port].url
