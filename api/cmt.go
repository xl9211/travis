package api

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"os"
	"strconv"
	"time"

	"github.com/spf13/cast"

	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	ethState "github.com/ethereum/go-ethereum/core/state"
	ethTypes "github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/rlp"
	"github.com/ethereum/go-ethereum/trie"
	abci "github.com/tendermint/tendermint/abci/types"
	cmn "github.com/tendermint/tendermint/libs/common"
	tmcmn "github.com/tendermint/tendermint/libs/common"
	"github.com/tendermint/tendermint/rpc/core"
	ctypes "github.com/tendermint/tendermint/rpc/core/types"
	ttypes "github.com/tendermint/tendermint/types"

	"github.com/CyberMiles/travis/modules/governance"
	"github.com/CyberMiles/travis/modules/stake"
	"github.com/CyberMiles/travis/sdk"
	"github.com/CyberMiles/travis/types"
	"github.com/CyberMiles/travis/utils"
)

// CmtRPCService offers cmt related RPC methods
type CmtRPCService struct {
	backend   *Backend
	am        *accounts.Manager
	nonceLock *AddrLocker
}

func NewCmtRPCService(b *Backend, nonceLock *AddrLocker) *CmtRPCService {
	return &CmtRPCService{
		backend:   b,
		am:        b.ethereum.AccountManager(),
		nonceLock: nonceLock,
	}
}

// sign tx and broadcast commit to tendermint.
func (s *CmtRPCService) signAndBroadcastTxCommit(args *SendTxArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	signTx := func(args *SendTxArgs) (*ethTypes.Transaction, error) {
		if args.Nonce == nil {
			// Hold the addresse's mutex around signing to prevent concurrent assignment of
			// the same nonce to multiple accounts.
			s.nonceLock.LockAddr(args.From)
			// release noncelock after sign
			defer s.nonceLock.UnlockAddr(args.From)
		}
		return s.backend.signTransaction(args)
	}

	signed, err := signTx(args)
	if err != nil {
		return nil, err
	}

	return s.backend.BroadcastTxCommit(signed)
}

// SendRawTx will broadcast the signed transaction to tendermint.
// The sender is responsible for signing the transaction and using the correct nonce.
func (s *CmtRPCService) SendRawTx(encodedTx hexutil.Bytes) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := new(ethTypes.Transaction)
	if err := rlp.DecodeBytes(encodedTx, tx); err != nil {
		return nil, err
	}

	if utils.IsEthTx(tx) {
		result, err := s.backend.BroadcastTxSync(tx)
		if err != nil {
			return nil, err
		}
		if result.Code > 0 {
			return nil, errors.New(result.Log)
		}

		return &ctypes.ResultBroadcastTxCommit{
			Hash: ttypes.Tx(encodedTx).Hash(), //tx.Hash().Hex(),
		}, nil
	} else {
		return s.backend.BroadcastTxCommit(tx)
	}
}

// GetBlockByNumber returns the requested block by height.
func (s *CmtRPCService) GetBlockByNumber(height uint64, decodeTx bool) (*ctypes.ResultBlock, error) {
	h := cast.ToInt64(height)
	block, err := s.backend.GetLocalClient().Block(&h)
	if err != nil {
		return nil, err
	}
	if !decodeTx {
		return block, err
	}
	// decode txs
	formatTx := func(index int) ([]byte, error) {
		// get transaction by hash
		tx := ttypes.Tx(block.Block.Txs[index])
		rpcTx, err := s.GetTransactionByHash(hex.EncodeToString(tx.Hash()))
		b, err := json.Marshal(rpcTx)
		return b, err
	}

	txs := block.Block.Txs
	for i := range txs {
		if txs[i], err = formatTx(i); err != nil {
			return nil, err
		}
	}
	return block, nil
}

func (s *CmtRPCService) DumpAllAddress() (string, error) {
	go dumpAllAddressCore(s)
	return "Dump doing", nil
}

func dumpAllAddressCore(s *CmtRPCService) {
	fmt.Printf("VULCANLABS dumpAllAddressCore begin...\n")
	bc := s.backend.Ethereum().BlockChain()
	state, err := bc.State()
	if err != nil {
		fmt.Printf("VULCANLABS State error: %v\n", err)
	}
	tempTrie, err := state.Database().OpenTrie(bc.CurrentBlock().Root())
	if err != nil {
		fmt.Printf("VULCANLABS OpenTrie error: %v\n", err)
	}

	addresses := make([]string, 0)
	it := trie.NewIterator(tempTrie.NodeIterator(nil))
	for it.Next() {
		addr := tempTrie.GetKey(it.Key)
		addresses = append(addresses, common.Bytes2Hex(addr))
	}

	filePath := fmt.Sprintf("./addresses-%d.txt", time.Now().Unix())
	writeArrayToFile(addresses, filePath)

	fmt.Printf("VULCANLABS dumpAllAddressCore end...\n")
}

func writeArrayToFile(addresses []string, filePath string) {
	fmt.Printf("VULCANLABS writeArrayToFile begin...\n")
	f, err := os.Create(filePath)
	if err != nil {
		fmt.Printf("VULCANLABS Create file error: %v\n", err)
		return
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, address := range addresses {
		lineStr := fmt.Sprintf("%s", address)
		fmt.Fprintln(w, lineStr)
	}
	w.Flush()
	fmt.Printf("VULCANLABS writeArrayToFile end...\n")
}

func (s *CmtRPCService) GetAllAddress() (string, error) {
	bc := s.backend.Ethereum().BlockChain()
	state, err := bc.State()
	if err != nil {
		return "", err
	}
	tempTrie, err := state.Database().OpenTrie(bc.CurrentBlock().Root())
	if err != nil {
		return "", err
	}

	addresses := make([]string, 0)
	it := trie.NewIterator(tempTrie.NodeIterator(nil))
	for it.Next() {
		addr := tempTrie.GetKey(it.Key)
		addresses = append(addresses, common.Bytes2Hex(addr))
	}

	data, _ := json.Marshal(addresses)
	dataString := string(data)

	return dataString, nil
}

func (s *CmtRPCService) DumpRawDataCore() (string, error) {
	go dumpRawDataCore(s)
	return "Dump doing", nil
}

func dumpRawDataCore(s *CmtRPCService) {
	fmt.Printf("VULCANLABS dumpRawDataCore begin...\n")
	bc := s.backend.Ethereum().BlockChain()
	state, err := bc.State()
	if err != nil {
		fmt.Printf("VULCANLABS State error: %v\n", err)
	}
	tempTrie, err := state.Database().OpenTrie(bc.CurrentBlock().Root())
	if err != nil {
		fmt.Printf("VULCANLABS OpenTrie error: %v\n", err)
	}

	data := make(map[string]ethState.DumpAccount)
	it := trie.NewIterator(tempTrie.NodeIterator(nil))
	for it.Next() {
		address := common.BytesToAddress(tempTrie.GetKey(it.Key))

		account := ethState.DumpAccount{
			Balance: state.GetBalance(address).String(),
			Nonce:   state.GetNonce(address),
			Code:    common.Bytes2Hex(state.GetCode(address)),
			Storage: make(map[string]string),
		}

		state.ForEachStorage(address, func(key, val common.Hash) bool {
			realVal := state.GetState(address, key)
			account.Storage[key.Hex()] = realVal.Hex()
			return true
		})

		data[address] = account
	}

	filePath := fmt.Sprintf("./rawdata-%d.txt", time.Now().Unix())
	writeMapToFile(data, filePath)

	fmt.Printf("VULCANLABS dumpRawDataCore end...\n")
}

func writeMapToFile(data map[string]ethState.DumpAccount, filePath string) {
	fmt.Printf("VULCANLABS writeMapToFile begin...\n")
	f, err := os.Create(filePath)
	if err != nil {
		fmt.Printf("VULCANLABS Create file error: %v\n", err)
		return
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for k, v := range data {
		data, _ := json.Marshal(v)
		dataString := string(data)
		lineStr := fmt.Sprintf("%x\t%s", k, dataString)
		fmt.Fprintln(w, lineStr)
	}
	w.Flush()
	fmt.Printf("VULCANLABS writeMapToFile end...\n")
}

func (s *CmtRPCService) GetAccountByAddress(address common.Address) (string, error) {
	state, err := s.backend.Ethereum().BlockChain().State()
	if err != nil {
		return "", err
	}

	account := ethState.DumpAccount{
		Balance: state.GetBalance(address).String(),
		Nonce:   state.GetNonce(address),
		Code:    common.Bytes2Hex(state.GetCode(address)),
		Storage: make(map[string]string),
	}

	state.ForEachStorage(address, func(key, val common.Hash) bool {
		realVal := state.GetState(address, key)
		account.Storage[key.Hex()] = realVal.Hex()
		return true
	})

	data, _ := json.Marshal(account)
	dataString := string(data)

	return dataString, nil
}

func (s *CmtRPCService) GetBalance(address common.Address) (string, error) {
	state, err := s.backend.Ethereum().BlockChain().State()
	if err != nil {
		return "", err
	}
	balance := state.GetBalance(address)
	return balance.String(), nil
}

func (s *CmtRPCService) GetCode(address common.Address) (string, error) {
	state, err := s.backend.Ethereum().BlockChain().State()
	if err != nil {
		return "", err
	}
	code := state.GetCode(address)
	return string(code), nil
}

func (s *CmtRPCService) GetNonce(address common.Address) (string, error) {
	state, err := s.backend.Ethereum().BlockChain().State()
	if err != nil {
		return "", err
	}
	nonce := state.GetNonce(address)
	return strconv.FormatUint(nonce, 10), nil
}

func (s *CmtRPCService) GetStorage1(address common.Address) (string, error) {
	state, err := s.backend.Ethereum().BlockChain().State()
	if err != nil {
		return "", err
	}

	storageTrie := state.StorageTrie(address)
	if storageTrie == nil {
		return "storage trie nil", err
	}

	storage := make(map[string]string)
	storageIt := trie.NewIterator(storageTrie.NodeIterator(nil))
	for storageIt.Next() {
		storage[common.Bytes2Hex(storageIt.Key)] = common.Bytes2Hex(storageIt.Value)
	}

	data, _ := json.Marshal(storage)
	dataString := string(data)

	return dataString, nil
}

func (s *CmtRPCService) GetStorage2(address common.Address) (string, error) {
	state, err := s.backend.Ethereum().BlockChain().State()
	if err != nil {
		return "", err
	}

	fmt.Printf("VULCANLABS Address: {%x}\n", address)
	storage := make(map[string]string)
	state.ForEachStorage(address, func(key, val common.Hash) bool {
		fmt.Printf("VULCANLABS key: %s value: %s\n", key.Hex(), val.Hex())
		storage[key.Hex()] = val.Hex()
		return true
	})

	fmt.Printf("VULCANLABS Result Address: {%x}\n", address)

	data, _ := json.Marshal(storage)
	dataString := string(data)

	fmt.Printf("VULCANLABS data: %s\n", dataString)

	return dataString, nil
}

func (s *CmtRPCService) GetStorage3(address common.Address) (string, error) {
	state, err := s.backend.Ethereum().BlockChain().State()
	if err != nil {
		return "", err
	}

	fmt.Printf("VULCANLABS Address: {%x}\n", address)
	storage := make(map[string]string)
	state.ForEachStorage(address, func(key, val common.Hash) bool {
		fmt.Printf("VULCANLABS key: %s value: %s\n", key.Hex(), val.Hex())
		realVal := state.GetState(address, key)
		fmt.Printf("VULCANLABS key: %s value: %s real value: %s\n", key.Hex(), val.Hex(), realVal.Hex())
		storage[key.Hex()] = realVal.Hex()
		return true
	})

	fmt.Printf("VULCANLABS Result Address: {%x}\n", address)

	data, _ := json.Marshal(storage)
	dataString := string(data)

	fmt.Printf("VULCANLABS data: %s\n", dataString)

	return dataString, nil
}

// RPCTransaction represents a transaction that will serialize to the RPC representation of a transaction
type RPCTransaction struct {
	BlockNumber      *hexutil.Big           `json:"blockNumber"`
	From             common.Address         `json:"from"`
	Gas              hexutil.Uint64         `json:"gas"`
	GasPrice         *hexutil.Big           `json:"gasPrice"`
	Hash             common.Hash            `json:"hash"`
	CmtHash          tmcmn.HexBytes         `json:"cmtHash"`
	Input            hexutil.Bytes          `json:"input"`
	CmtInput         interface{}            `json:"cmtInput"`
	Nonce            hexutil.Uint64         `json:"nonce"`
	To               *common.Address        `json:"to"`
	TransactionIndex hexutil.Uint           `json:"transactionIndex"`
	Value            *hexutil.Big           `json:"value"`
	V                *hexutil.Big           `json:"v"`
	R                *hexutil.Big           `json:"r"`
	S                *hexutil.Big           `json:"s"`
	TxResult         abci.ResponseDeliverTx `json:"txResult"`
}

// newRPCTransaction returns a transaction that will serialize to the RPC representation.
func newRPCTransaction(res *ctypes.ResultTx) (*RPCTransaction, error) {
	tx := new(ethTypes.Transaction)
	rlpStream := rlp.NewStream(bytes.NewBuffer(res.Tx), 0)
	if err := tx.DecodeRLP(rlpStream); err != nil {
		return nil, err
	}

	var travisTx sdk.Tx
	if !utils.IsEthTx(tx) {
		if err := json.Unmarshal(tx.Data(), &travisTx); err != nil {
			return nil, err
		}
	}
	blockNumber := uint64(res.Height)
	index := uint64(res.Index)
	rpcTx := newEthRPCTransaction(tx, blockNumber, index)
	if rpcTx != nil {
		rpcTx.CmtHash = res.Hash
		rpcTx.CmtInput = travisTx
		rpcTx.TxResult = res.TxResult
	}

	return rpcTx, nil
}

// GetTransactionFromBlock returns the transaction for the given block number and index.
func (s *CmtRPCService) GetTransactionFromBlock(height uint64, index uint64) (*RPCTransaction, error) {
	// get block
	h := cast.ToInt64(height)
	block, err := s.backend.GetLocalClient().Block(&h)
	if err != nil {
		return nil, err
	}
	// check index
	if cast.ToInt64(index) >= block.Block.NumTxs {
		return nil, errors.New(fmt.Sprintf("No transaction in block %d, index %d. ", height, index))
	}
	// get transaction by hash
	tx := ttypes.Tx(block.Block.Txs[index])
	return s.GetTransactionByHash(hex.EncodeToString(tx.Hash()))
}

// GetTransactionByHash returns the transaction for the given hash
func (s *CmtRPCService) GetTransactionByHash(hash string) (*RPCTransaction, error) {
	// bytes from hash string
	bkey, err := hex.DecodeString(cmn.StripHex(hash))
	if err != nil {
		return nil, err
	}
	// get transaction
	res, err := s.backend.GetLocalClient().Tx(bkey, false)
	if err == nil {
		return newRPCTransaction(res)
	}
	errNotFound := fmt.Sprintf("Tx (%X) not found", bkey)
	if err != nil && err.Error() != errNotFound {
		// error other than not found
		return nil, err
	}
	// No finalized transaction, try to retrieve it from the pool
	unConfirmedTxs, err := core.UnconfirmedTxs(-1)
	if err != nil {
		return nil, err
	}
	for _, tx := range unConfirmedTxs.Txs {
		if bytes.Equal(bkey, tx.Hash()) {
			rpcTx, err := newRPCTransaction(&ctypes.ResultTx{Tx: ttypes.Tx(tx), Hash: tx.Hash()})
			if err != nil {
				return nil, err
			}
			return rpcTx, nil
		}
	}
	// Transaction unknown, return as such
	return nil, err
}

// DecodeRawTxs returns the transactions from the raw tx array in the block data
func (s *CmtRPCService) DecodeRawTxs(rawTxs []string) ([]*RPCTransaction, error) {
	txs := make([]*RPCTransaction, len(rawTxs))
	for index, raw := range rawTxs {
		rawTx, err := base64.StdEncoding.DecodeString(raw)
		if err != nil {
			return txs, err
		}
		tx := ttypes.Tx(rawTx)
		rpcTx, err := newRPCTransaction(&ctypes.ResultTx{Tx: tx, Hash: tx.Hash()})
		if err != nil {
			return txs, err
		}
		txs[index] = rpcTx
	}
	return txs, nil
}

// Info about the node's syncing state
func (s *CmtRPCService) Syncing() (*ctypes.SyncInfo, error) {
	status, err := s.backend.GetLocalClient().Status()
	if err != nil {
		return nil, err
	}

	return &status.SyncInfo, nil
}

// Get number of unconfirmed transactions in the mempool.
func (s *CmtRPCService) PendingTransactionCount() (*hexutil.Uint64, error) {
	res, err := core.NumUnconfirmedTxs()
	if err != nil {
		return nil, err
	}
	num := uint64(res.N)
	return (*hexutil.Uint64)(&num), nil
}

// Get unconfirmed transactions in the mempool(limit default=30, max=100).
func (s *CmtRPCService) GetPendingTransactions(limit int) ([]*RPCTransaction, error) {
	unConfirmedTxs, err := core.UnconfirmedTxs(limit)
	if err != nil {
		return nil, err
	}
	if unConfirmedTxs.N == 0 {
		return nil, nil
	}

	txs := make([]*RPCTransaction, len(unConfirmedTxs.Txs))
	for index, tx := range unConfirmedTxs.Txs {
		rpcTx, err := newRPCTransaction(&ctypes.ResultTx{Tx: tx, Hash: tx.Hash()})
		if err != nil {
			return txs, err
		}
		txs[index] = rpcTx
	}
	return txs, nil
}

func (s *CmtRPCService) makeTravisTxArgs(tx sdk.Tx, address common.Address, nonce *hexutil.Uint64) (*SendTxArgs, error) {
	data, err := tx.MarshalJSON()
	if err != nil {
		return nil, err
	}

	zeroUint := hexutil.Uint64(0)
	zeroBigInt := big.NewInt(0)
	return &SendTxArgs{
		address,
		nil,
		&zeroUint,
		(*hexutil.Big)(zeroBigInt),
		(*hexutil.Big)(zeroBigInt),
		nonce,
		(*hexutil.Bytes)(&data),
		nil,
	}, nil
}

type DeclareCandidacyArgs struct {
	Nonce       *hexutil.Uint64   `json:"nonce"`
	From        common.Address    `json:"from"`
	PubKey      string            `json:"pubKey"`
	MaxAmount   hexutil.Big       `json:"maxAmount"`
	CompRate    sdk.Rat           `json:"compRate"`
	Description stake.Description `json:"description"`
}

func (s *CmtRPCService) DeclareCandidacy(args DeclareCandidacyArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	pubKey, err := types.GetPubKey(args.PubKey)
	if err != nil {
		return nil, err
	}
	tx := stake.NewTxDeclareCandidacy(pubKey, args.MaxAmount.ToInt().String(), args.CompRate, args.Description)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type WithdrawCandidacyArgs struct {
	Nonce *hexutil.Uint64 `json:"nonce"`
	From  common.Address  `json:"from"`
}

func (s *CmtRPCService) WithdrawCandidacy(args WithdrawCandidacyArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxWithdrawCandidacy()

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type UpdateCandidacyArgs struct {
	Nonce       *hexutil.Uint64   `json:"nonce"`
	From        common.Address    `json:"from"`
	PubKey      string            `json:"pubKey"`
	MaxAmount   *hexutil.Big      `json:"maxAmount"`
	CompRate    sdk.Rat           `json:"compRate"`
	Description stake.Description `json:"description"`
}

func (s *CmtRPCService) UpdateCandidacy(args UpdateCandidacyArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	maxAmount := ""
	if args.MaxAmount != nil {
		maxAmount = args.MaxAmount.ToInt().String()
	}

	pubKey := types.PubKey{}
	if !utils.IsBlank(args.PubKey) {
		tmp, err := types.GetPubKey(args.PubKey)
		if err != nil {
			return nil, err
		}
		pubKey = tmp
	}

	tx := stake.NewTxUpdateCandidacy(pubKey, maxAmount, args.CompRate, args.Description)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type ActivateCandidacyArgs struct {
	Nonce *hexutil.Uint64 `json:"nonce"`
	From  common.Address  `json:"from"`
}

func (s *CmtRPCService) ActivateCandidacy(args ActivateCandidacyArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxActivateCandidacy()

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type DeactivateCandidacyArgs struct {
	Nonce *hexutil.Uint64 `json:"nonce"`
	From  common.Address  `json:"from"`
}

func (s *CmtRPCService) DeactivateCandidacy(args DeactivateCandidacyArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxDeactivateCandidacy()

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type SetCompRateArgs struct {
	Nonce            *hexutil.Uint64 `json:"nonce"`
	From             common.Address  `json:"from"`
	DelegatorAddress common.Address  `json:"delegatorAddress"`
	CompRate         sdk.Rat         `json:"compRate"`
}

func (s *CmtRPCService) SetCompRate(args SetCompRateArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxSetCompRate(args.DelegatorAddress, args.CompRate)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type UpdateCandidacyAccountArgs struct {
	Nonce               *hexutil.Uint64 `json:"nonce"`
	From                common.Address  `json:"from"`
	NewCandidateAddress common.Address  `json:"newCandidateAccount"`
}

func (s *CmtRPCService) UpdateCandidacyAccount(args UpdateCandidacyAccountArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxUpdateCandidacyAccount(args.NewCandidateAddress)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type AcceptCandidacyAccountUpdateArgs struct {
	Nonce                  *hexutil.Uint64 `json:"nonce"`
	From                   common.Address  `json:"from"`
	AccountUpdateRequestId int64           `json:"accountUpdateRequestId"`
}

func (s *CmtRPCService) AcceptCandidacyAccountUpdate(args AcceptCandidacyAccountUpdateArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxAcceptCandidacyAccountUpdate(args.AccountUpdateRequestId)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type VerifyCandidacyArgs struct {
	Nonce            *hexutil.Uint64 `json:"nonce"`
	From             common.Address  `json:"from"`
	CandidateAddress common.Address  `json:"candidateAddress"`
	Verified         bool            `json:"verified"`
}

func (s *CmtRPCService) VerifyCandidacy(args VerifyCandidacyArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxVerifyCandidacy(args.CandidateAddress, args.Verified)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type DelegateArgs struct {
	Nonce            *hexutil.Uint64 `json:"nonce"`
	From             common.Address  `json:"from"`
	ValidatorAddress common.Address  `json:"validatorAddress"`
	Amount           hexutil.Big     `json:"amount"`
	CubeBatch        string          `json:"cubeBatch"`
	Sig              string          `json:"sig"`
}

func (s *CmtRPCService) Delegate(args DelegateArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxDelegate(args.ValidatorAddress, args.Amount.ToInt().String(), args.CubeBatch, args.Sig, "")

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type WithdrawArgs struct {
	Nonce              *hexutil.Uint64 `json:"nonce"`
	From               common.Address  `json:"from"`
	ValidatorAddress   common.Address  `json:"validatorAddress"`
	Amount             hexutil.Big     `json:"amount"`
	CompletelyWithdraw bool            `json:"completelyWithdraw"`
}

func (s *CmtRPCService) Withdraw(args WithdrawArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := stake.NewTxWithdraw(args.ValidatorAddress, args.Amount.ToInt().String(), args.CompletelyWithdraw)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type StakeQueryResult struct {
	Height int64       `json:"height"`
	Data   interface{} `json:"data"`
}

func (s *CmtRPCService) QueryValidators(height uint64) (*StakeQueryResult, error) {
	var candidates stake.Candidates
	h, err := s.getParsedFromJson("/validators", []byte{0}, &candidates, height)
	if err != nil {
		return nil, err
	}

	return &StakeQueryResult{h, candidates}, nil
}

func (s *CmtRPCService) QueryValidator(address common.Address, height uint64) (*StakeQueryResult, error) {
	var candidate stake.Candidate
	h, err := s.getParsedFromJson("/validator", []byte(address.Hex()), &candidate, height)
	if err != nil {
		return nil, err
	}

	return &StakeQueryResult{h, &candidate}, nil
}

func (s *CmtRPCService) QueryDelegator(address common.Address, height uint64) (*StakeQueryResult, error) {
	var slotDelegates []*stake.Delegation
	h, err := s.getParsedFromJson("/delegator", []byte(address.Hex()), &slotDelegates, height)
	if err != nil {
		return nil, err
	}

	return &StakeQueryResult{h, slotDelegates}, nil
}

func (s *CmtRPCService) QueryAwardInfos(height uint64) (*StakeQueryResult, error) {
	var awardInfos stake.AwardInfos
	h, err := s.getParsedFromJson("/awardInfo", utils.AwardInfosKey, &awardInfos, height)
	if err != nil {
		return nil, err
	}

	return &StakeQueryResult{h, awardInfos}, nil
}

func (s *CmtRPCService) QueryAbsentValidators(height uint64) (*StakeQueryResult, error) {
	var absentValidators stake.AbsentValidators
	h, err := s.getParsedFromJson("/key", utils.AbsentValidatorsKey, &absentValidators, height)
	if err != nil {
		return nil, err
	}

	return &StakeQueryResult{h, absentValidators}, nil
}

type GovernanceTransferFundProposalArgs struct {
	Nonce             *hexutil.Uint64 `json:"nonce"`
	From              common.Address  `json:"from"`
	TransferFrom      common.Address  `json:"transferFrom"`
	TransferTo        common.Address  `json:"transferTo"`
	Amount            hexutil.Big     `json:"amount"`
	Reason            string          `json:"reason"`
	ExpireTimestamp   *int64          `json:"expireTimestamp"`
	ExpireBlockHeight *int64          `json:"expireBlockHeight"`
}

func (s *CmtRPCService) ProposeTransferFund(args GovernanceTransferFundProposalArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := governance.NewTxTransferFundPropose(&args.TransferFrom, &args.TransferTo,
		args.Amount.ToInt().String(), args.Reason,
		args.ExpireTimestamp, args.ExpireBlockHeight)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type GovernanceChangeParamProposalArgs struct {
	Nonce             *hexutil.Uint64 `json:"nonce"`
	From              common.Address  `json:"from"`
	Name              string          `json:"name"`
	Value             string          `json:"value"`
	Reason            string          `json:"reason"`
	ExpireTimestamp   *int64          `json:"expireTimestamp"`
	ExpireBlockHeight *int64          `json:"expireBlockHeight"`
}

func (s *CmtRPCService) ProposeChangeParam(args GovernanceChangeParamProposalArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := governance.NewTxChangeParamPropose(args.Name, args.Value, args.Reason,
		args.ExpireTimestamp, args.ExpireBlockHeight)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type GovernanceDeployLibEniProposalArgs struct {
	Nonce             *hexutil.Uint64 `json:"nonce"`
	From              common.Address  `json:"from"`
	Name              string          `json:"name"`
	Version           string          `json:"version"`
	FileUrl           string          `json:"fileUrl"`
	Md5               string          `json:"md5"`
	Reason            string          `json:"reason"`
	DeployTimestamp   *int64          `json:"deployTimestamp"`
	DeployBlockHeight *int64          `json:"deployBlockHeight"`
}

func (s *CmtRPCService) ProposeDeployLibEni(args GovernanceDeployLibEniProposalArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := governance.NewTxDeployLibEniPropose(args.Name, args.Version, args.FileUrl, args.Md5, args.Reason,
		args.DeployTimestamp, args.DeployBlockHeight)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type GovernanceRetireProgramProposalArgs struct {
	Nonce               *hexutil.Uint64 `json:"nonce"`
	From                common.Address  `json:"from"`
	PreservedValidators string          `json:"preservedValidators"`
	Reason              string          `json:"reason"`
	RetiredBlockHeight  *int64          `json:"retiredBlockHeight"`
}

func (s *CmtRPCService) ProposeRetireProgram(args GovernanceRetireProgramProposalArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := governance.NewTxRetireProgramPropose(args.PreservedValidators, args.Reason, args.RetiredBlockHeight)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type GovernanceUpgradeProgramProposalArgs struct {
	Nonce              *hexutil.Uint64 `json:"nonce"`
	From               common.Address  `json:"from"`
	Name               string          `json:"name"`
	Version            string          `json:"version"`
	FileUrl            string          `json:"fileUrl"`
	Md5                string          `json:"md5"`
	Reason             string          `json:"reason"`
	UpgradeBlockHeight *int64          `json:"upgradeBlockHeight"`
}

func (s *CmtRPCService) ProposeUpgradeProgram(args GovernanceUpgradeProgramProposalArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := governance.NewTxUpgradeProgramPropose(args.Name,
		args.Version, args.FileUrl, args.Md5, args.Reason, args.UpgradeBlockHeight)

	txArgs, err := s.makeTravisTxArgs(tx, args.From, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

type GovernanceVoteArgs struct {
	Nonce      *hexutil.Uint64 `json:"nonce"`
	Voter      common.Address  `json:"from"`
	ProposalId string          `json:"proposalId"`
	Answer     string          `json:"answer"`
}

func (s *CmtRPCService) Vote(args GovernanceVoteArgs) (*ctypes.ResultBroadcastTxCommit, error) {
	tx := governance.NewTxVote(args.ProposalId, args.Answer)

	txArgs, err := s.makeTravisTxArgs(tx, args.Voter, args.Nonce)
	if err != nil {
		return nil, err
	}

	return s.signAndBroadcastTxCommit(txArgs)
}

func (s *CmtRPCService) QueryProposals() (*StakeQueryResult, error) {
	var proposals []*governance.Proposal
	h, err := s.getParsedFromJson("/governance/proposals", []byte{0}, &proposals, 0)
	if err != nil {
		return nil, err
	}

	return &StakeQueryResult{h, proposals}, nil
}

func (s *CmtRPCService) QueryParams(height uint64) (*StakeQueryResult, error) {
	var params utils.Params
	h, err := s.getParsedFromJson("/key", utils.ParamKey, &params, height)
	if err != nil {
		return nil, err
	}
	return &StakeQueryResult{h, params}, nil
}
