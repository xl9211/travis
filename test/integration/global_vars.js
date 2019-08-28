module.exports = {
  TestMode: "cluster",

  Accounts: [],
  PubKeys: [
    "051FUvSNJmVL4UiFL7ucBr3TnGqG6a5JgUIgKf4UOIA=",
    "v0yMKq/chUKEhELdLp1HJfGAmHZJll8cEeskU5L97Mg=",
    "lmlbeRtIZLSgIvib9Emndk/W0isuGrJmBDlB+EwbYuY=",
    "GzGGwxzBnEj8RbFFAMgH+QP8bPsyRXrlTknpkt8mo5o="
  ],
  CubeBatch: "01",
  Params: {},
  ETH: {
    bytecode: "",
    abi: [],
    contractAddress: "b6b29ef90120bec597939e0eda6b8a9164f75deb"
  },
  Reverse: {
    bytecode:
      "0x608060405234801561001057600080fd5b50610272806100206000396000f300608060405260043610610041576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff168063064767aa14610046575b600080fd5b34801561005257600080fd5b506100ad600480360381019080803590602001908201803590602001908080601f0160208091040260200160405190810160405280939291908181526020018383808284378201915050505050509192919290505050610128565b6040518080602001828103825283818151815260200191508051906020019080838360005b838110156100ed5780820151818401526020810190506100d2565b50505050905090810190601f16801561011a5780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b60608060405160206040519081016040526001905260206040519081016040527f0400000000000000000000000000000000000000000000000000000000000000905260206040519081016040526001905260206040519081016040527f04000000000000000000000000000000000000000000000000000000000000009052602060405190810160405280600090528480516020019081604051908101604052905b6020831015156101f057805182526020820191506020810190506020830392506101cb565b6001836020036101000a0d8019825116818451168082178552505050505050806040518190039052907f7265766572736500000000000000000000000000000000000000000000000000f59050809150509190505600a165627a7a723058204272fd3392dd08f71aad18d99b3593526f5b5b8859012881cab51740081dd8360029",
    abi: [
      {
        constant: false,
        inputs: [{ name: "input", type: "string" }],
        name: "reverse",
        outputs: [{ name: "", type: "string" }],
        payable: false,
        stateMutability: "nonpayable",
        type: "function"
      }
    ],
    contractAddress: ""
  },
  JsonTest: {
    bytecode:
        "0x608060405234801561001057600080fd5b506117dc806100206000396000f300608060405260043610610083576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff16806302384aa714610088578063090da5ce1461016a57806350d84b761461024c57806371ebe6b11461032e57806373172725146104105780638c2dc1c6146104f2578063f1b46219146105d4575b600080fd5b34801561009457600080fd5b506100ef600480360381019080803590602001908201803590602001908080601f01602080910402602001604051908101604052809392919081815260200183838082843782019150505050505091929192905050506106b6565b6040518080602001828103825283818151815260200191508051906020019080838360005b8381101561012f578082015181840152602081019050610114565b50505050905090810190601f16801561015c5780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b34801561017657600080fd5b506101d1600480360381019080803590602001908201803590602001908080601f0160208091040260200160405190810160405280939291908181526020018383808284378201915050505050509192919290505050610850565b6040518080602001828103825283818151815260200191508051906020019080838360005b838110156102115780820151818401526020810190506101f6565b50505050905090810190601f16801561023e5780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b34801561025857600080fd5b506102b3600480360381019080803590602001908201803590602001908080601f0160208091040260200160405190810160405280939291908181526020018383808284378201915050505050509192919290505050610880565b6040518080602001828103825283818151815260200191508051906020019080838360005b838110156102f35780820151818401526020810190506102d8565b50505050905090810190601f1680156103205780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b34801561033a57600080fd5b50610395600480360381019080803590602001908201803590602001908080601f0160208091040260200160405190810160405280939291908181526020018383808284378201915050505050509192919290505050610912565b6040518080602001828103825283818151815260200191508051906020019080838360005b838110156103d55780820151818401526020810190506103ba565b50505050905090810190601f1680156104025780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b34801561041c57600080fd5b50610477600480360381019080803590602001908201803590602001908080601f0160208091040260200160405190810160405280939291908181526020018383808284378201915050505050509192919290505050610942565b6040518080602001828103825283818151815260200191508051906020019080838360005b838110156104b757808201518184015260208101905061049c565b50505050905090810190601f1680156104e45780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b3480156104fe57600080fd5b50610559600480360381019080803590602001908201803590602001908080601f01602080910402602001604051908101604052809392919081815260200183838082843782019150505050505091929192905050506109ae565b6040518080602001828103825283818151815260200191508051906020019080838360005b8381101561059957808201518184015260208101905061057e565b50505050905090810190601f1680156105c65780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b3480156105e057600080fd5b5061063b600480360381019080803590602001908201803590602001908080601f0160208091040260200160405190810160405280939291908181526020018383808284378201915050505050509192919290505050610a2d565b6040518080602001828103825283818151815260200191508051906020019080838360005b8381101561067b578082015181840152602081019050610660565b50505050905090810190601f1680156106a85780820d805160018360200d6101000a0d1916815260200191505b509250505060405180910390f35b60606106c0611796565b6106c983610aa4565b9050600060058111156106d857fe5b610700846106f286600086610ad39092919063ffffffff16565b610c7a90919063ffffffff16565b600581111561070b57fe5b14151561071757600080fd5b6001600581111561072457fe5b61074c8461073e86600186610ad39092919063ffffffff16565b610c7a90919063ffffffff16565b600581111561075757fe5b14151561076357600080fd5b6002600581111561077057fe5b6107988461078a86600286610ad39092919063ffffffff16565b610c7a90919063ffffffff16565b60058111156107a357fe5b1415156107af57600080fd5b600360058111156107bc57fe5b6107e4846107d686600386610ad39092919063ffffffff16565b610c7a90919063ffffffff16565b60058111156107ef57fe5b1415156107fb57600080fd5b6004600581111561080857fe5b6108308461082286600486610ad39092919063ffffffff16565b610c7a90919063ffffffff16565b600581111561083b57fe5b14151561084757600080fd5b82915050919050565b606061085a611796565b61086383610aa4565b9050610878838261110f90919063ffffffff16565b915050919050565b606061088a611796565b61089383610aa4565b9050600115156108c1846108b386600086610ad39092919063ffffffff16565b61129a90919063ffffffff16565b15151415156108cf57600080fd5b600015156108fb846108ed86600186610ad39092919063ffffffff16565b61129a90919063ffffffff16565b151514151561090957600080fd5b82915050919050565b606061091c611796565b61092583610aa4565b905061093a838261110f90919063ffffffff16565b915050919050565b606061094c611796565b610954611796565b606061095f85610aa4565b925061097785600185610ad39092919063ffffffff16565b91506109a18561099387600086610ad39092919063ffffffff16565b61110f90919063ffffffff16565b9050809350505050919050565b60606109b8611796565b60606109c384610aa4565b9150610a2184610a13866040805190810160405280600581526020017f68656c6c6f000000000000000000000000000000000000000000000000000000815250866113459092919063ffffffff16565b61110f90919063ffffffff16565b90508092505050919050565b6060610a37611796565b610a3f611796565b610a4884610aa4565b9150610a6084600184610ad39092919063ffffffff16565b9050600a8160000151141515610a7557600080fd5b60118160200151141515610a8857600080fd5b610a9b848261110f90919063ffffffff16565b92505050919050565b610aac611796565b610ab4611796565b6000816000018181525050825181602001818152505080915050919050565b610adb611796565b6000808560000151915085602001519050610c6f60405160206040519081016040526005905260206040519081016040527f0404090909000000000000000000000000000000000000000000000000000000905260206040519081016040526001905260206040519081016040527f04000000000000000000000000000000000000000000000000000000000000009052602060405190810160405280600090526040805190810160405280600890526020017f61727261794765740000000000000000000000000000000000000000000000008152508780516020019081601f0160209004602002604051908101604052905b602083101515610bf45780518252602082019150602081019050602083039250610bcf565b6001836020036101000a0d801982511681845116808217855250505050505084602060405190810160405252836020604051908101604052528660206040519081016040525280604051819003602090039052907f6a736f6e00000000000000000000000000000000000000000000000000000000f5611540565b925050509392505050565b60007f7400000000000000000000000000000000000000000000000000000000000000828460000151815181101515610caf57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff19161480610dc557507f6600000000000000000000000000000000000000000000000000000000000000828460000151815181101515610d5657fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916145b15610dd35760009050611109565b818360000151815181101515610de557fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f0100000000000000000000000000000000000000000000000000000000000000900460ff16603011158015610ee957506039828460000151815181101515610e7557fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f0100000000000000000000000000000000000000000000000000000000000000900460ff1611155b15610ef75760019050611109565b7f2200000000000000000000000000000000000000000000000000000000000000828460000151815181101515610f2a57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff19161415610fa65760029050611109565b7f5b00000000000000000000000000000000000000000000000000000000000000828460000151815181101515610fd957fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191614156110555760039050611109565b7f7b0000000000000000000000000000000000000000000000000000000000000082846000015181518110151561108857fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191614156111045760049050611109565b600590505b92915050565b6060600080846000015191508460200151905060405160206040519081016040526004905260206040519081016040527f0404090900000000000000000000000000000000000000000000000000000000905260206040519081016040526001905260206040519081016040527f04000000000000000000000000000000000000000000000000000000000000009052602060405190810160405280600090526040805190810160405280600b90526020017f7061727365537472696e670000000000000000000000000000000000000000008152508580516020019081601f0160209004602002604051908101604052905b6020831015156112275780518252602082019150602081019050602083039250611202565b6001836020036101000a0d8019825116818451168082178552505050505050836020604051908101604052528260206040519081016040525280604051819003602090039052907f6a736f6e00000000000000000000000000000000000000000000000000000000f59250505092915050565b60007f74000000000000000000000000000000000000000000000000000000000000008284600001518151811015156112cf57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191614905092915050565b61134d611796565b600080856000015191508560200151905061153560405160206040519081016040526005905260206040519081016040527f0404090904000000000000000000000000000000000000000000000000000000905260206040519081016040526001905260206040519081016040527f04000000000000000000000000000000000000000000000000000000000000009052602060405190810160405280600090526040805190810160405280600990526020017f6f626a65637447657400000000000000000000000000000000000000000000008152508780516020019081601f0160209004602002604051908101604052905b6020831015156114665780518252602082019150602081019050602083039250611441565b6001836020036101000a0d801982511681845116808217855250505050505084602060405190810160405252836020604051908101604052528680516020019081601f0160209004602002604051908101604052905b6020831015156114e157805182526020820191506020810190506020830392506114bc565b6001836020036101000a0d801982511681845116808217855250505050505080604051819003602090039052907f6a736f6e00000000000000000000000000000000000000000000000000000000f5611540565b925050509392505050565b611548611796565b61155382600061155a565b9050919050565b611562611796565b61156a611796565b611574848461159d565b81600001818152505061158a846008850161159d565b8160200181815250508091505092915050565b6000806000809150600884019350600090505b602081101561178b5783806001900394505060397f01000000000000000000000000000000000000000000000000000000000000000285858151811015156115f457fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff19161115156116f657806030868681518110151561167957fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f01000000000000000000000000000000000000000000000000000000000000009004039060020a0282179150611780565b806057868681518110151561170757fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f01000000000000000000000000000000000000000000000000000000000000009004039060020a02821791505b6004810190506115b0565b819250505092915050565b6040805190810160405280600081526020016000815250905600a165627a7a72305820f1aab3c2c0ef6bafaae66317547f2ff9ae978aa8cf65c408d35ce54d3672b5bf0029",
    abi: [
      {
        "constant": true,
        "inputs": [
          {
            "name": "raw",
            "type": "string"
          }
        ],
        "name": "testGetType",
        "outputs": [
          {
            "name": "",
            "type": "string"
          }
        ],
        "payable": false,
        "stateMutability": "pure",
        "type": "function"
      },
      {
        "constant": true,
        "inputs": [
          {
            "name": "raw",
            "type": "string"
          }
        ],
        "name": "testAsString",
        "outputs": [
          {
            "name": "",
            "type": "string"
          }
        ],
        "payable": false,
        "stateMutability": "pure",
        "type": "function"
      },
      {
        "constant": true,
        "inputs": [
          {
            "name": "raw",
            "type": "string"
          }
        ],
        "name": "testAsBool",
        "outputs": [
          {
            "name": "",
            "type": "string"
          }
        ],
        "payable": false,
        "stateMutability": "pure",
        "type": "function"
      },
      {
        "constant": true,
        "inputs": [
          {
            "name": "raw",
            "type": "string"
          }
        ],
        "name": "testAsInt",
        "outputs": [
          {
            "name": "",
            "type": "string"
          }
        ],
        "payable": false,
        "stateMutability": "pure",
        "type": "function"
      },
      {
        "constant": true,
        "inputs": [
          {
            "name": "raw",
            "type": "string"
          }
        ],
        "name": "testNestedArrayGet",
        "outputs": [
          {
            "name": "",
            "type": "string"
          }
        ],
        "payable": false,
        "stateMutability": "pure",
        "type": "function"
      },
      {
        "constant": true,
        "inputs": [
          {
            "name": "raw",
            "type": "string"
          }
        ],
        "name": "testObjectGet",
        "outputs": [
          {
            "name": "",
            "type": "string"
          }
        ],
        "payable": false,
        "stateMutability": "pure",
        "type": "function"
      },
      {
        "constant": true,
        "inputs": [
          {
            "name": "raw",
            "type": "string"
          }
        ],
        "name": "testArrayGet",
        "outputs": [
          {
            "name": "",
            "type": "string"
          }
        ],
        "payable": false,
        "stateMutability": "pure",
        "type": "function"
      }
    ],
    contractAddress: ""
  },
  Dogecoin: {
    bytecode:
      "0x608060405234801561001057600080fd5b506111d5806100206000396000f300608060405260043610610041576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff168063a83dc30614610046575b600080fd5b34801561005257600080fd5b5061015760048036038101908080359060200190929190803590602001908201803590602001908080601f0160208091040260200160405190810160405280939291908181526020018383808284378201915050505050509192919290803590602001908201803590602001908080601f016020809104026020016040519081016040528093929190818152602001838380828437820191505050505050919291929080359060200190929190803590602001908201803590602001908080601f016020809104026020016040519081016040528093929190818152602001838380828437820191505050505050919291929080359060200190929190505050610171565b604051808215151515815260200191505060405180910390f35b600061017b611172565b606080600060c0604051908101604052808c81526020018b81526020018a81526020018981526020018881526020018781525093506101b98461030c565b92506102d360405160206040519081016040526001905260206040519081016040527f0400000000000000000000000000000000000000000000000000000000000000905260206040519081016040526001905260206040519081016040527f04000000000000000000000000000000000000000000000000000000000000009052602060405190810160405280600090528580516020019081604051908101604052905b602083101515610283578051825260208201915060208101905060208303925061025e565b6001836020036101000a0d8019825116818451168082178552505050505050806040518190039052907f7363727970740000000000000000000000000000000000000000000000000000f561082f565b91506102de876109ea565b9050806102ea83610a2e565b11156102f957600094506102fe565b600194505b505050509695505050505050565b60608060608060608060608060008060a06040519080825280601f01601f19166020018201604052801561034f5781602001602082028038833980820191505090505b5098506103696103648c600001516008610da2565b61082f565b97506103788b6020015161082f565b96506103878b6040015161082f565b95506103a061039b8c606001516008610da2565b61082f565b94506103af8b6080015161082f565b93506103c86103c38c60a001516008610da2565b61082f565b925060009050600091505b87518210156104865787828151811015156103ea57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f010000000000000000000000000000000000000000000000000000000000000002898280600101935081518110151561044957fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a90535081806001019250506103d3565b600091505b865182101561053e5786828151811015156104a257fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f010000000000000000000000000000000000000000000000000000000000000002898280600101935081518110151561050157fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a905350818060010192505061048b565b600091505b85518210156105f657858281518110151561055a57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f01000000000000000000000000000000000000000000000000000000000000000289828060010193508151811015156105b957fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a9053508180600101925050610543565b600091505b84518210156106ae57848281518110151561061257fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f010000000000000000000000000000000000000000000000000000000000000002898280600101935081518110151561067157fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a90535081806001019250506105fb565b600091505b83518210156107665783828151811015156106ca57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f010000000000000000000000000000000000000000000000000000000000000002898280600101935081518110151561072957fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a90535081806001019250506106b3565b600091505b825182101561081e57828281518110151561078257fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f01000000000000000000000000000000000000000000000000000000000000000289828060010193508151811015156107e157fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a905350818060010192505061076b565b889950505050505050505050919050565b6060806060600084925060006002845181151561084857fe5b0614151561085257fe5b82516040519080825280601f01601f1916602001820160405280156108865781602001602082028038833980820191505090505b509150600090505b82518110156109df5782600282855103038151811015156108ab57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f010000000000000000000000000000000000000000000000000000000000000002828281518110151561090457fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a905350826001828551030381518110151561094857fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f01000000000000000000000000000000000000000000000000000000000000000282600183018151811015156109a457fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a90535060028101905061088e565b819350505050919050565b6000806000806109f985610a2e565b9250601863ff00000084169060020a900491506008600383030291508162ffffff84169060020a029050809350505050919050565b6000606060008084925060009050600091505b8251821015610d97576004819060020a0290507f30000000000000000000000000000000000000000000000000000000000000007f0100000000000000000000000000000000000000000000000000000000000000900460ff168383815181101515610aa957fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f0100000000000000000000000000000000000000000000000000000000000000900460ff1610158015610bec57507f39000000000000000000000000000000000000000000000000000000000000007f0100000000000000000000000000000000000000000000000000000000000000900460ff168383815181101515610b7857fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f0100000000000000000000000000000000000000000000000000000000000000900460ff1611155b15610cbe577f30000000000000000000000000000000000000000000000000000000000000007f010000000000000000000000000000000000000000000000000000000000000090048383815181101515610c4357fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f010000000000000000000000000000000000000000000000000000000000000090040360ff1681019050610d8a565b600a7f61000000000000000000000000000000000000000000000000000000000000007f010000000000000000000000000000000000000000000000000000000000000090048484815181101515610d1257fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f0100000000000000000000000000000000000000000000000000000000000000027f01000000000000000000000000000000000000000000000000000000000000009004030160ff16810190505b8180600101925050610a41565b809350505050919050565b60606000606060006060604080519080825280601f01601f191660200182016040528015610ddf5781602001602082028038833980820191505090505b509250600093505b604084101561107057600f87167f010000000000000000000000000000000000000000000000000000000000000002915060007f010000000000000000000000000000000000000000000000000000000000000002827effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191610158015610eb2575060097f010000000000000000000000000000000000000000000000000000000000000002827effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191611155b15610f8857817f010000000000000000000000000000000000000000000000000000000000000090047f30000000000000000000000000000000000000000000000000000000000000007f01000000000000000000000000000000000000000000000000000000000000009004017f0100000000000000000000000000000000000000000000000000000000000000028385603f03815181101515610f5357fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a905350611058565b600a827f010000000000000000000000000000000000000000000000000000000000000090047f61000000000000000000000000000000000000000000000000000000000000007f0100000000000000000000000000000000000000000000000000000000000000900401037f0100000000000000000000000000000000000000000000000000000000000000028385603f0381518110151561102757fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a9053505b6004879060020a900496508380600101945050610de7565b8560ff166040519080825280601f01601f1916602001820160405280156110a65781602001602082028038833980820191505090505b509050600093505b8560ff168410156111655782848760400360ff16018151811015156110cf57fe5b9060200101517f010000000000000000000000000000000000000000000000000000000000000090047f010000000000000000000000000000000000000000000000000000000000000002818581518110151561112857fe5b9060200101907effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916908160001a90535083806001019450506110ae565b8094505050505092915050565b60c06040519081016040528060008152602001606081526020016060815260200160008152602001606081526020016000815250905600a165627a7a7230582038802dc5aad6a913b855eb4cf2e625aea15168883684c587e2621db18efee4540029",
    abi: [
      {
        constant: true,
        inputs: [
          { name: "version", type: "uint256" },
          { name: "prev_block", type: "string" },
          { name: "merkle_root", type: "string" },
          { name: "timestamp", type: "uint256" },
          { name: "bits", type: "string" },
          { name: "nonce", type: "uint256" }
        ],
        name: "verifyBlock",
        outputs: [{ name: "", type: "bool" }],
        payable: false,
        stateMutability: "pure",
        type: "function"
      }
    ],
    contractAddress: ""
  },
  FreeGas: {
    bytecode:
      "0x608060405234801561001057600080fd5b50610346806100206000396000f30060806040526004361061004c576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff16806303be5ce91461004e578063e615f35b1461008f575b005b34801561005a57600080fd5b50610079600480360381019080803590602001909291905050506100d0565b6040518082815260200191505060405180910390f35b34801561009b57600080fd5b506100ba600480360381019080803590602001909291905050506101a2565b6040518082815260200191505060405180910390f35b600060606000606060405190810160405280603481526020017f6162636465666768696a6b6c6d6e6f707172737475767778797a61626364656681526020017f6768696a6b6c6d6e6f707172737475767778797a0000000000000000000000008152509150600090505b600a81101561018f576001829080600181540180825580915050906001820390600052602060002001600090919290919091509080519060200190610180929190610275565b5050808060010191505061013a565b8360008190555060005492505050919050565b600060606000606060405190810160405280603481526020017f6162636465666768696a6b6c6d6e6f707172737475767778797a61626364656681526020017f6768696a6b6c6d6e6f707172737475767778797a0000000000000000000000008152509150600090505b600a811015610261576001829080600181540180825580915050906001820390600052602060002001600090919290919091509080519060200190610252929190610275565b5050808060010191505061020c565b8360008190555060005492505050919050f8565b82805460018160011615610100020d166002900490600052602060002090601f016020900481019282601f106102b657805160ff19168380011785556102e4565b828001600101855582156102e4579182015b828111156102e35782518255916020019190600101906102c8565b5b5090506102f191906102f5565b5090565b61031791905b808211156103135760008160009055506001016102fb565b5090565b905600a165627a7a72305820367ac56c14359bb405c4f27dec3226655841ea92708331c635862ceca737e7ee0029",
    abi: [
      {
        constant: false,
        inputs: [{ name: "input", type: "int256" }],
        name: "testNonFreeGas",
        outputs: [{ name: "", type: "int256" }],
        payable: false,
        stateMutability: "nonpayable",
        type: "function"
      },
      {
        constant: false,
        inputs: [{ name: "input", type: "int256" }],
        name: "testFreeGas",
        outputs: [{ name: "", type: "int256" }],
        payable: false,
        stateMutability: "nonpayable",
        type: "function"
      },
      {
        payable: true,
        stateMutability: "payable",
        type: "fallback"
      }
    ],
    contractAddress: ""
  },
  LibEni: {
    FileUrl:
      '{"ubuntu":["https://libeni.cybermiles.io/libs/reverse/eni_reverse_1.2.0_ubuntu16.04.so", "http://34.85.18.42:8000/eni_reverse_1.2.0_ubuntu16.04.so"], \
        "centos":["https://libeni.cybermiles.io/libs/reverse/eni_reverse_1.2.0_centos7.so", "http://34.85.18.42:8000/eni_reverse_1.2.0_centos7.so"]}',
    MD5:
      '{"ubuntu":"b440ff88be3fb2d47da4f5b34577d92477bb7f01b52d9d3a09557ea83c97a696211453ff75fb3446b1e99e1a520df2d6539b47bc5151f2689598ecbba23e906d", \
        "centos":"04ae4cd328bd550aae2c18f9fb2945ab849ec763a075f2d6e6010a676dba526082233722827d684a0de733c48b7faa2846094026657d42c3bf360a313c7b0851"}'
  },

  GasFeeKey: Buffer.from("GasFee").toString("base64")
}
