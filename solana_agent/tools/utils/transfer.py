import logging
import based58
from solana.rpc.commitment import Confirmed, Finalized
from solders.transaction import Transaction, VersionedTransaction
from solders.pubkey import Pubkey
from solders.message import Message, to_bytes_versioned
from solders.compute_budget import set_compute_unit_limit
from solders.system_program import TransferParams, transfer
from solders.null_signer import NullSigner
from solders.instruction import Instruction
from solders.keypair import Keypair
from spl.token.async_client import AsyncToken
from spl.token.instructions import (
    transfer_checked as spl_transfer,
    TransferCheckedParams as SPLTransferParams,
    create_associated_token_account,
    get_associated_token_address,
)
from solana_agent.tools.utils.wallet import SolanaWalletClient

LAMPORTS_PER_SOL = 10**9
SPL_TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"


def make_memo_instruction(memo: str) -> Instruction:  # pragma: no cover
    return Instruction(
        program_id=Pubkey.from_string(MEMO_PROGRAM_ID),
        accounts=[],
        data=memo.encode("utf-8"),
    )


class TokenTransferManager:
    @staticmethod
    async def _is_valid_ata_owner(
        wallet: SolanaWalletClient, owner_pubkey: Pubkey
    ) -> bool:
        try:
            if not owner_pubkey.is_on_curve():
                return False
            info = await wallet.client.get_account_info(owner_pubkey)
            value = getattr(info, "value", None)
            if not value:
                return True
            return str(value.owner) == SYSTEM_PROGRAM_ID
        except Exception:
            # Never block transfers due to RPC/account lookup issues.
            return True

    @staticmethod
    async def transfer(  # pragma: no cover
        wallet: SolanaWalletClient,
        to: str,
        amount: float,
        mint: str,
        provider: str = None,
        no_signer: bool = False,
        fee_percentage: float = 0.0,
        memo: str = "",
    ) -> Transaction:
        """
        Transfer SOL, SPL, or Token2022 tokens to a recipient.

        :param wallet: An instance of SolanaWalletClient
        :param to: Recipient's public key
        :param amount: Amount to transfer
        :param mint: Optional mint address for SPL or Token2022 token
        :param provider: Provider for the transaction, default is None
        :param no_signer: If True, doesn't sign the transaction with the wallet's keypair
        :param fee_percentage: Optional percentage of the transfer amount to be used as a fee (default 0.0 = no fees)
        :param memo: Optional memo for the transaction
        :return: Transaction object ready for submission
        """
        try:
            # Convert to PublicKey objects
            to_pubkey = Pubkey.from_string(to)
            wallet_pubkey = wallet.pubkey
            wallet_keypair = wallet.keypair

            fee_payer_keypair = getattr(wallet, "fee_payer", None)
            tx_payer_pubkey = (
                fee_payer_keypair.pubkey()
                if (no_signer and fee_payer_keypair)
                else wallet_pubkey
            )

            if mint == "So11111111111111111111111111111111111111112":
                ixs = []
                ix_transfer = transfer(
                    TransferParams(
                        from_pubkey=wallet_pubkey,
                        to_pubkey=to_pubkey,
                        lamports=int(amount * LAMPORTS_PER_SOL),
                    )
                )
                ixs.append(ix_transfer)

                if memo:
                    ix_memo = make_memo_instruction(memo)
                    ixs.append(ix_memo)

                if fee_payer_keypair and fee_percentage > 0:
                    fee_lamports = int(
                        amount * LAMPORTS_PER_SOL * (fee_percentage / 100)
                    )
                    if fee_lamports > 0:
                        ix_fee = transfer(
                            TransferParams(
                                from_pubkey=wallet_pubkey,
                                to_pubkey=fee_payer_keypair.pubkey(),
                                lamports=fee_lamports,
                            )
                        )
                        ixs.append(ix_fee)

                if no_signer:
                    blockhash_response = await wallet.client.get_latest_blockhash(
                        commitment=Finalized,
                    )
                    recent_blockhash = blockhash_response.value.blockhash
                    msg = Message.new_with_blockhash(
                        instructions=ixs,
                        payer=tx_payer_pubkey,
                        blockhash=recent_blockhash,
                    )

                    msg_bytes = to_bytes_versioned(msg)
                    num_signers = msg.header.num_required_signatures
                    signer_keys = list(msg.account_keys)[:num_signers]
                    signatures = [
                        NullSigner(signer_pk).sign_message(msg_bytes)
                        for signer_pk in signer_keys
                    ]

                    if fee_payer_keypair:
                        fee_payer_pubkey = fee_payer_keypair.pubkey()
                        for i, signer_pk in enumerate(signer_keys):
                            if signer_pk == fee_payer_pubkey:
                                signatures[i] = fee_payer_keypair.sign_message(
                                    msg_bytes
                                )
                                break

                    return VersionedTransaction.populate(
                        message=msg,
                        signatures=signatures,
                    )

                blockhash_response = await wallet.client.get_latest_blockhash(
                    commitment=Finalized,
                )
                recent_blockhash = blockhash_response.value.blockhash

                msg = Message(
                    instructions=ixs,
                    payer=wallet_pubkey,
                )

                transaction = Transaction(
                    from_keypairs=[wallet_keypair],
                    message=msg,
                    recent_blockhash=recent_blockhash,
                )

                cu_units = (
                    await wallet.client.simulate_transaction(
                        transaction, commitment=Confirmed
                    )
                ).value.units_consumed

                compute_budget_ix = set_compute_unit_limit(int(cu_units + 100_000))

                new_msg = Message(
                    instructions=[*ixs, compute_budget_ix],
                    payer=wallet_pubkey,
                )

                blockhash_response = await wallet.client.get_latest_blockhash(
                    commitment=Finalized,
                )
                recent_blockhash = blockhash_response.value.blockhash

                new_transaction = Transaction(
                    from_keypairs=[wallet_keypair],
                    message=new_msg,
                    recent_blockhash=recent_blockhash,
                )

                if provider == "helius":
                    encoded_transaction = based58.b58encode(
                        bytes(new_transaction), based58.Alphabet.DEFAULT
                    ).decode("utf-8")

                    priority_fee = await wallet.get_priority_fee_estimate_helius(
                        encoded_transaction
                    )

                    new_transaction.message.instructions.insert(
                        0,
                        set_compute_unit_limit(priority_fee),
                    )

                return new_transaction

            else:
                mint_pubkey = Pubkey.from_string(mint)
                resp = await wallet.client.get_account_info(mint_pubkey)
                owner = str(resp.value.owner)
                if owner == SPL_TOKEN_PROGRAM_ID:
                    program_id = Pubkey.from_string(SPL_TOKEN_PROGRAM_ID)
                elif owner == TOKEN_2022_PROGRAM_ID:
                    program_id = Pubkey.from_string(TOKEN_2022_PROGRAM_ID)
                else:
                    raise ValueError(
                        f"Unsupported token program: {owner}. Supported programs are SPL Token and Token 2022."
                    )

                token_payer = fee_payer_keypair or wallet_keypair or Keypair()
                token = AsyncToken(wallet.client, mint_pubkey, program_id, token_payer)

                ixs = []

                from_ata = (
                    (await token.get_accounts_by_owner(wallet_pubkey)).value[0].pubkey
                )

                to_ata = get_associated_token_address(
                    to_pubkey, mint_pubkey, token_program_id=program_id
                )

                # Check if the *specific* destination ATA exists.
                # NOTE: get_accounts_by_owner can return non-ATA token accounts;
                # we always transfer to the canonical ATA, so we must ensure it exists.
                to_ata_info = await wallet.client.get_account_info(to_ata)
                if not getattr(to_ata_info, "value", None):
                    create_ata_ix = create_associated_token_account(
                        payer=tx_payer_pubkey,
                        owner=to_pubkey,
                        mint=mint_pubkey,
                        token_program_id=program_id,
                    )
                    ixs.append(create_ata_ix)

                mint_info = await token.get_mint_info()
                adjusted_amount = int(amount * (10**mint_info.decimals))

                ix_spl = spl_transfer(
                    SPLTransferParams(
                        program_id=program_id,
                        source=from_ata,
                        mint=mint_pubkey,
                        dest=to_ata,
                        owner=wallet_pubkey,
                        amount=adjusted_amount,
                        decimals=mint_info.decimals,
                    )
                )
                ixs.append(ix_spl)

                if fee_payer_keypair and fee_percentage > 0:
                    fee_amount = int(
                        amount * (10**mint_info.decimals) * (fee_percentage / 100)
                    )

                    # Fee payer may not yet have an ATA for this mint; create it if needed.
                    # Also avoid generating a 0-amount token transfer (common for dust amounts).
                    if fee_amount > 0:
                        fee_payer_pubkey = fee_payer_keypair.pubkey()

                        # Some deployments configure `fee_payer` to a non-wallet address
                        # (e.g., a token account). The associated token program rejects
                        # creating ATAs for non-system-owned owners, which would fail the
                        # entire transfer. Treat token fee collection as best-effort.
                        if not await TokenTransferManager._is_valid_ata_owner(
                            wallet, fee_payer_pubkey
                        ):
                            logging.warning(
                                "Skipping token fee collection: fee_payer is not a valid ATA owner"
                            )
                        else:
                            to_fee_ata = get_associated_token_address(
                                fee_payer_pubkey,
                                mint_pubkey,
                                token_program_id=program_id,
                            )

                            to_fee_ata_info = await wallet.client.get_account_info(
                                to_fee_ata
                            )
                            if not getattr(to_fee_ata_info, "value", None):
                                create_fee_ata_ix = create_associated_token_account(
                                    payer=tx_payer_pubkey,
                                    owner=fee_payer_pubkey,
                                    mint=mint_pubkey,
                                    token_program_id=program_id,
                                )
                                ixs.append(create_fee_ata_ix)

                            ix_fee = spl_transfer(
                                SPLTransferParams(
                                    program_id=program_id,
                                    source=from_ata,
                                    mint=mint_pubkey,
                                    dest=to_fee_ata,
                                    owner=wallet_pubkey,
                                    amount=fee_amount,
                                    decimals=mint_info.decimals,
                                )
                            )
                            ixs.append(ix_fee)

                if memo:
                    ix_memo = make_memo_instruction(memo)
                    ixs.append(ix_memo)

                if no_signer:
                    blockhash_response = await wallet.client.get_latest_blockhash(
                        commitment=Finalized,
                    )
                    recent_blockhash = blockhash_response.value.blockhash
                    msg = Message.new_with_blockhash(
                        instructions=ixs,
                        payer=tx_payer_pubkey,
                        blockhash=recent_blockhash,
                    )

                    msg_bytes = to_bytes_versioned(msg)
                    num_signers = msg.header.num_required_signatures
                    signer_keys = list(msg.account_keys)[:num_signers]
                    signatures = [
                        NullSigner(signer_pk).sign_message(msg_bytes)
                        for signer_pk in signer_keys
                    ]

                    if fee_payer_keypair:
                        fee_payer_pubkey = fee_payer_keypair.pubkey()
                        for i, signer_pk in enumerate(signer_keys):
                            if signer_pk == fee_payer_pubkey:
                                signatures[i] = fee_payer_keypair.sign_message(
                                    msg_bytes
                                )
                                break

                    return VersionedTransaction.populate(
                        message=msg,
                        signatures=signatures,
                    )

                blockhash_response = await wallet.client.get_latest_blockhash(
                    commitment=Finalized,
                )
                recent_blockhash = blockhash_response.value.blockhash

                msg = Message(
                    instructions=ixs,
                    payer=wallet_pubkey,
                )

                transaction = Transaction(
                    from_keypairs=[wallet_keypair],
                    message=msg,
                    recent_blockhash=recent_blockhash,
                )

                cu_units = (
                    await wallet.client.simulate_transaction(
                        transaction, commitment=Confirmed
                    )
                ).value.units_consumed

                compute_budget_ix = set_compute_unit_limit(int(cu_units + 100_000))

                new_msg = Message(
                    instructions=[*ixs, compute_budget_ix],
                    payer=wallet_pubkey,
                )

                blockhash_response = await wallet.client.get_latest_blockhash(
                    commitment=Finalized,
                )
                recent_blockhash = blockhash_response.value.blockhash

                new_transaction = Transaction(
                    from_keypairs=[wallet_keypair],
                    message=new_msg,
                    recent_blockhash=recent_blockhash,
                )

                if provider == "helius":
                    encoded_transaction = based58.b58encode(
                        bytes(new_transaction), based58.Alphabet.DEFAULT
                    ).decode("utf-8")

                    priority_fee = await wallet.get_priority_fee_estimate_helius(
                        encoded_transaction
                    )

                    new_transaction.message.instructions.insert(
                        0,
                        set_compute_unit_limit(priority_fee),
                    )
                return new_transaction

        except Exception as e:
            logging.exception(f"Transfer failed: {str(e)}")
            raise RuntimeError(f"Transfer failed: {str(e)}")
